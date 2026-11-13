"""MMMU-Pro vision against ENDPOINT with the A2 check's sampling (temperature
1.0, top-p 0.95, up to 65,536 new tokens, 64 concurrent), run by lmms-eval
through the A2 check's client, scored by score.py.

PROMPT=lmms sends lmms-eval's mmmu_pro_vision request, as the A2 check does:
the image and an empty text. PROMPT=official adds MMMU-Pro's own vision
prompt (direct setting) as the text. Prints one JSON line and keeps it as
RUN_DIR/result.json.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["HARNESS_CHECKS"])

import score  # noqa: E402
from mmstar import env, failed_requests, run_lmms_eval  # noqa: E402

# MMMU-Pro's vision prompt, direct setting (MMMU-Benchmark/MMMU, mmmu-pro/prompts.yaml).
OFFICIAL_PROMPT = (
    "Answer with the option letter from the given choices directly. The last "
    "line of your response should be of the following format: 'Answer: $LETTER' "
    "(without quotes) where LETTER is one of options."
)


def official_task_dir():
    """lmms-eval's mmmu_pro_vision, renamed, with the official prompt as its text."""
    import lmms_eval

    src = Path(lmms_eval.__file__).parent / "tasks" / "mmmu_pro"
    dst = Path(env("RUN_DIR"), "tasks", "mmmu_pro_vision_prompt")
    dst.mkdir(parents=True)
    shutil.copy(src / "_default_template_yaml", dst)
    utils = (src / "utils.py").read_text()
    utils += (
        "\n\ndef official_vision_prompt(doc, lmms_eval_specific_kwargs=None):\n"
        f"    return {OFFICIAL_PROMPT!r}\n"
    )
    (dst / "utils.py").write_text(utils)
    task = (src / "mmmu_pro_vision.yaml").read_text()
    task = re.sub(r"tag:\n(  - .*\n)+", "", task)
    task = task.replace('task: "mmmu_pro_vision"', 'task: "mmmu_pro_vision_prompt"')
    task = task.replace("utils.mmmu_pro_doc_to_text", "utils.official_vision_prompt")
    (dst / "mmmu_pro_vision_prompt.yaml").write_text(task)
    return dst.parent


def main():
    prompt = os.environ.get("PROMPT", "lmms")
    if prompt == "official":
        task, include = "mmmu_pro_vision_prompt", official_task_dir()
    else:
        task, include = "mmmu_pro_vision", None
    _, samples = run_lmms_eval(
        "mmmu_pro_vision",
        [task],
        "temperature=1.0,max_new_tokens=65536",
        64,
        include,
        extra={"top_p": 0.95},
    )
    lines = samples[task]
    result = {
        "variant": env("VARIANT"),
        "prompt": prompt,
        **score.score(lines),
        "failed_requests": failed_requests(lines),
        "run_dir": env("RUN_DIR"),
    }
    line = json.dumps(result)
    Path(env("RUN_DIR"), "result.json").write_text(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
