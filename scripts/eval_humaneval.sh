#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:?Usage: bash scripts/eval_humaneval.sh <model_or_adapter_path> [extra args]}"
shift || true

cd "${PROJECT_ROOT}"
python "${PROJECT_ROOT}/scripts/eval_codegen.py" \
  --model "${MODEL_PATH}" \
  --benchmark humaneval \
  "$@"

