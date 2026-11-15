---
repo: junyyang-amd/ATOM
integration_branch: kimi-k3-tpot
remote: fork
never_touch: ROCm/ATOM
design_entry: CLAUDE.md
gate_task: black --check . && bash .github/scripts/run_unit_tests.sh && cd atom/mesh && cargo test -- --test-threads=1
new_tests_dir: tests
publish_language: English
max_tasks: 3
---

# ATOM on the kimi-k3-tpot line

The work tracked here is wuhuikx/LLM-Booster#351: why Kimi-K3 decode TPOT is
4-5% higher on `rocm/atom-dev:ROCm_10.1.0` than on the ROCm 7.2.4 image
`rocm/atom-dev:nightly_202611021458`. The line starts at
`junyyang/rocm-10.1-hip7.16` (ROCm/ATOM#2901, 7ba27eefe). ROCm/ATOM is
upstream: agents never push to it, comment on it or open PRs there.
Agents never post to wuhuikx/LLM-Booster either; the owner does.

- This is an investigation: most work is measurement on GPU nodes, and its
  output is scripts, trace summaries and a report under `docs/`, listed
  from `CLAUDE.md`. Raw traces stay on `/share_nfs`, never in git.
- The gate mirrors the PR CI (black, the CPU unit suite, the ATOMesh Rust
  tests) and needs no GPU. Ruff runs on changed lines only, as CI's
  reviewdog does.
- GPU work runs only on pit2-p03-g07, g08 and g24, in the two images
  above; how to reach them is in the local file. A node whose GPUs are busy
  with someone else's job is a stop: escalate, never kill another user's
  container.
- The project skills `capture-trace`, `run-atom-workload` and
  `debug-agent-locate-kernel` under `.claude/skills/` cover tracing and
  launching a workload.
