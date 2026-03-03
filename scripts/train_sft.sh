#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${PROJECT_ROOT}/config/qwen2_5_oft_sft.yaml}"
shift || true

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "${PROJECT_ROOT}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun \
    --nproc_per_node "${NUM_GPUS}" \
    --master_port "${MASTER_PORT}" \
    "${PROJECT_ROOT}/scripts/train_sft.py" \
    "${CONFIG_PATH}" \
    "$@"
else
  python "${PROJECT_ROOT}/scripts/train_sft.py" "${CONFIG_PATH}" "$@"
fi

