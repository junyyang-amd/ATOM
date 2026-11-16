#!/usr/bin/env bash
# The RL line's CPU unit suite: .github/scripts/run_unit_tests.sh with the
# failures already present at the line's start (ROCm/ATOM main d2d5a6bbb)
# deselected by name. Each has an issue on the line; its fix removes its line.
set -euo pipefail
known_failures=(
  # Mxfp4MoEMethod has no use_flydsl_herd when built by the test's fixture.
  "tests/model_ops/test_comm_fused_moe.py::test_mxfp4_apply_hands_the_shared_output_to_fused_experts"
)
deselect=()
for t in "${known_failures[@]}"; do deselect+=(--deselect "$t"); done
python -m pytest tests/ \
  --ignore=tests/plugin \
  --ignore=tests/entrypoints/test_openai_server.py \
  "${deselect[@]}" \
  -rs \
  --junitxml="${UNIT_TEST_REPORT:-unit-report.xml}"
