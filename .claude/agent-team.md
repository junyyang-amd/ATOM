---
repo: junyyang-amd/ATOM
integration_branch: EPD
remote: fork
never_touch: ROCm/ATOM
design_entry: CLAUDE.md
gate_task: black --check . && bash .github/scripts/run_unit_tests.sh && cd atom/mesh && cargo test -- --test-threads=1
new_tests_dir: tests
publish_language: English
max_tasks: 3
---

# ATOM on the EPD line

The work tracked here is wuhuikx/LLM-Booster#455: multimodal image requests
through ATOMesh and E/P/D disaggregation. ROCm/ATOM is upstream: agents never
push to it, comment on it or open PRs there; the owner upstreams from `EPD`.

- The gate mirrors the PR CI: `pre-checks.yaml` (black, ruff, the CPU unit
  suite) plus the ATOMesh Rust tests, which upstream runs only in GPU
  workflows. Nothing in the gate needs a GPU.
- Ruff is not in the portable gate: upstream `main` is not ruff-clean, and
  CI checks only changed lines (`ruff check --exit-zero` through reviewdog,
  `-filter-mode=diff_context`). A change adds no ruff finding on the lines
  it touches; the local gate checks exactly that.
- GPU work (smoke runs, accuracy, the acceptance harness) runs on MI355X
  nodes over ssh, in the model recipe's container; how to reach them is in
  the local file. A node whose GPUs are busy with someone else's job is a
  stop: escalate, never kill another user's container.
- Design documents go in `docs/` and are listed from `CLAUDE.md`.
