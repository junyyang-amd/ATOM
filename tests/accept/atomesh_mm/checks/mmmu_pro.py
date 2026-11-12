"""MMMU-Pro standard (10 options) and vision against the deployment under
test, with the recipe's sampling (temperature 1.0, top-p 0.95, up to 65,536
new tokens), by lmms-eval.

Each answer is scored by strict letter match: the option letter lmms-eval's
multiple-choice extractor finds in the response must equal the target.
lmms-eval's own vision scorer compares the whole response to the letter.
Passes when each subset's score is within 2 x its standard error of the
model profile's reference (MMMU_PRO_REF_STANDARD, MMMU_PRO_REF_VISION).
"""

import json

from mmstar import env, failed_requests, finish, run_check, run_lmms_eval, stderr

SUBSETS = {
    "standard": "mmmu_pro_standard",
    "vision": "mmmu_pro_vision",
}
LETTERS = list("ABCDEFGHIJ")


def letter_accuracy(lines):
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    samples = [json.loads(line) for line in lines]
    correct = 0
    for sample in samples:
        response = sample["filtered_resps"]
        if isinstance(response, list):
            response = response[0]
        correct += extract_mcq_answer(response, LETTERS) == sample["target"]
    return correct / len(samples)


def main():
    _, samples = run_lmms_eval(
        "mmmu_pro",
        list(SUBSETS.values()),
        "temperature=1.0,max_new_tokens=65536",
        64,
        extra={"top_p": 0.95},
    )
    result, failures = {}, []
    for subset, task in SUBSETS.items():
        n = len(samples[task])
        score = letter_accuracy(samples[task])
        ref = float(env(f"MMMU_PRO_REF_{subset.upper()}"))
        tolerance = 2 * stderr(score, n)
        result[subset] = {
            "score": score,
            "stderr": stderr(score, n),
            "n": n,
            "failed_requests": failed_requests(samples[task]),
            "reference": ref,
        }
        if abs(score - ref) > tolerance:
            failures.append(
                f"{subset} {score:.4f} is not within {tolerance:.4f} of {ref:.4f}"
            )
    finish(result, failures)


if __name__ == "__main__":
    run_check(main)
