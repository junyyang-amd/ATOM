"""MMMU-Pro standard (10 options) and vision against the deployment under
test, with the recipe's sampling (temperature 1.0, top-p 0.95, up to 65,536
new tokens), by lmms-eval.

Passes when each subset's score is within 2 x its standard error of the
model profile's reference (MMMU_PRO_REF_STANDARD, MMMU_PRO_REF_VISION).
"""

from mmstar import env, failed_requests, finish, run_check, run_lmms_eval, stderr

SUBSETS = {
    "standard": "mmmu_pro_standard",
    "vision": "mmmu_pro_vision",
}


def main():
    results, samples = run_lmms_eval(
        "mmmu_pro",
        list(SUBSETS.values()),
        "temperature=1.0,top_p=0.95,max_new_tokens=65536",
        64,
    )
    result, failures = {}, []
    for subset, task in SUBSETS.items():
        n = len(samples[task])
        score = results[task]["mmmu_acc,none"]
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
