#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec > >(tee run.log) 2>&1

PYTHON="${PYTHON:-.venv/bin/python}"

# Configure trials per dataset here (or override via env vars)
SMALL_TRIALS="${SMALL_TRIALS:-10000}"
MEDIUM_TRIALS="${MEDIUM_TRIALS:-10000}"
LARGE_TRIALS="${LARGE_TRIALS:-10000}"

run_one() {
  local size="$1"
  local trials="$2"
  echo "========== ${size} (trials=${trials}) =========="
  "${PYTHON}" -u project1.py "data/${size}.csv" "data/${size}.gph" "${trials}"
  echo
}

run_one small  "${SMALL_TRIALS}"
run_one medium "${MEDIUM_TRIALS}"
run_one large  "${LARGE_TRIALS}"

echo "========== scores =========="
for size in small medium large; do
  echo -n "${size}: "
  "${PYTHON}" -u project1.py "data/${size}.csv" "data/${size}.gph" --score
done
