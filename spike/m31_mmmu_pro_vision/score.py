"""Score MMMU-Pro samples (lmms-eval --log_samples jsonl) two ways.

- lmms: the A2 check's scorer, lmms-eval's extract_mcq_answer over the whole
  response (the last occurrence of the highest-priority letter format).
- answer_line: the letter of the last "Answer: X" / "answer is X" statement,
  else the lmms letter.

    python3 score.py <samples.jsonl>...   one JSON line per file
"""

import importlib.util
import json
import math
import os
import re
import sys

LETTERS = list("ABCDEFGHIJ")
ANSWER = re.compile(r"(?i:answer)\s*(?:is|:)\s*\**\s*\(?\s*\**([A-J])\b")


def _mcq():
    """lmms-eval's extractor, loaded by path so its package's imports are not needed."""
    root = os.environ.get("LMMS_EVAL_SRC", "/share_nfs/junyan_tmp_r120/accept/lmms-eval")
    path = f"{root}/lmms_eval/tasks/_task_utils/mcq_extract.py"
    spec = importlib.util.spec_from_file_location("mcq_extract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_mcq_answer


extract_mcq_answer = _mcq()


def lmms(response):
    return extract_mcq_answer(response, LETTERS)


def answer_line(response):
    found = ANSWER.findall(response)
    return found[-1] if found else lmms(response)


SCORERS = {"lmms": lmms, "answer_line": answer_line}


def response_of(sample):
    response = sample["filtered_resps"]
    return response[0] if isinstance(response, list) else response


def score(lines):
    samples = [json.loads(line) for line in lines if line.strip()]
    n = len(samples)
    result = {"n": n}
    for name, extract in SCORERS.items():
        acc = sum(extract(response_of(s)) == s["target"] for s in samples) / n
        result[name] = round(acc, 4)
        result[f"{name}_stderr"] = round(math.sqrt(acc * (1 - acc) / n), 4)
    out = [s["token_counts"][0]["output_tokens"] for s in samples if s.get("token_counts")]
    result["empty"] = sum(not response_of(s).strip() for s in samples)
    result["at_max_tokens"] = sum(o >= 65536 for o in out)
    return result


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with open(path) as f:
            print(json.dumps({"file": path, **score(f.readlines())}))
