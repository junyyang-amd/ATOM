#!/usr/bin/env bash
# The RL line's CPU unit suite: .github/scripts/run_unit_tests.sh with the
# failures already present at the line's start (ROCm/ATOM main d2d5a6bbb)
# deselected by name. Each has an issue on the line; its fix removes its line.
set -euo pipefail
known_failures=(
)
deselect=()
for t in "${known_failures[@]}"; do deselect+=(--deselect "$t"); done
python -m pytest tests/ \
  --ignore=tests/plugin \
  --ignore=tests/entrypoints/test_openai_server.py \
  "${deselect[@]}" \
  -rs \
  --junitxml="${UNIT_TEST_REPORT:-unit-report.xml}"
