"""Score one GSM8K run of run.sh: flexible-extract accuracy, failed requests.

Failed requests are the chat completions ATOMesh answered with a non-2xx
status, read from its log after the byte offset where the run started.
Empty responses are counted apart: lm-eval scores a request that failed
after its retries as an empty answer.
"""

import argparse
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STATUS = re.compile(r"uri=/v1/chat/completions .*status_code=(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--mesh-log", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rdma-rx-bytes", type=int, default=0)
    a = ap.parse_args()

    (results,) = a.results.rglob("results_*.json")
    gsm = json.loads(results.read_text())["results"]["gsm8k"]
    (samples,) = a.results.rglob("samples_gsm8k_*.jsonl")
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    rows = [r for r in rows if r.get("filter") == "flexible-extract"]
    empty = sum(1 for r in rows if not r["resps"][0][0].strip())

    with a.mesh_log.open("rb") as f:
        f.seek(a.offset)
        log = ANSI.sub("", f.read().decode(errors="replace"))
    codes = [int(m.group(1)) for m in STATUS.finditer(log)]
    failed = sum(1 for c in codes if not 200 <= c < 300)

    print(
        json.dumps(
            {
                "label": a.label,
                "n": len(rows),
                "flexible_extract": round(gsm["exact_match,flexible-extract"], 4),
                "stderr": round(gsm["exact_match_stderr,flexible-extract"], 4),
                "strict_match": round(gsm["exact_match,strict-match"], 4),
                "mesh_requests": len(codes),
                "failed_requests": failed,
                "empty_responses": empty,
                "rdma_rx_gib": round(a.rdma_rx_bytes / 2**30, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
