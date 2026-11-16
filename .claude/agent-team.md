---
repo: junyyang-amd/ATOM
integration_branch: RL
remote: fork
never_touch: ROCm/ATOM
design_entry: CLAUDE.md
gate_task: black --check . && bash tools/rl/unit_tests.sh && cd atom/mesh && cargo test -- --test-threads=1 --skip vendoring_rejects_unsafe_downloads_and_preserves_sources_on_failure
new_tests_dir: tests
publish_language: English
max_tasks: 3
---

# ATOM on the RL line

The work tracked here is wuhuikx/LLM-Booster#456: an RL rollout control
plane in ATOMesh (pause / refit / resume, `weight_version`) compatible with
SMG's `--enable-rl`, and the ATOM engine endpoints behind it. The line
starts at ROCm/ATOM main d2d5a6bbb. ROCm/ATOM is upstream: agents never
push to it, comment on it or open PRs there; the owner upstreams from
`RL`. Agents never post to wuhuikx/LLM-Booster either; the owner does.

- The gate mirrors the PR CI (black, the CPU unit suite, the ATOMesh Rust
  tests) and needs no GPU. `tools/rl/unit_tests.sh` runs the suite as
  `.github/scripts/run_unit_tests.sh` does, minus the failures present at
  the line's start, each named there with its issue; the one Rust failure
  at the start is the `--skip` in `gate_task`, also with its issue.
  Ruff runs on changed lines only, as CI's reviewdog does (`--exit-zero`, `diff_context`).
- GPU work (SGLang and ATOM workers, the acceptance harness
  `tools/rl/accept.py`) runs on MI355X nodes over ssh; how to reach them is
  in the local file. A node whose GPUs are busy with someone else's job is
  a stop: escalate, never kill another user's container.
- The SMG reference is smg-project/smg@6c7bda118c5c6c02e65fe104cc8d93a0f906c965
  (`crates/rl`, `examples/rl`). Read it from there; vendor none of it.
- Changing ATOM's weight loading (`atom/model_loader`, `atom/rollout`) or
  its scheduling core (`atom/model_engine`) needs no owner confirmation:
  the owner ruled on 2026-11-16 that it goes through design review like any
  other change, overriding #456's stop rule on it. #456's other stop rules
  hold.
- Design documents go in `docs/` and are listed from `CLAUDE.md`.
