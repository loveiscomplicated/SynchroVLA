#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-1601}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/dynamic_2x2_convergence}"
DATASET_PATH="${DATASET_PATH:-${OUTPUT_ROOT}/demos/dynamic_pick_place_demos.pt}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2 3}"

mkdir -p "${OUTPUT_ROOT}/logs"

# shellcheck disable=SC2086
"${PYTHON_BIN}" scripts/run_dynamic_2x2_gpu_seed.py \
  --seed "${SEED}" \
  --python-bin "${PYTHON_BIN}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-path "${DATASET_PATH}" \
  --device "${TRAIN_DEVICE}" \
  --eval-device "${EVAL_DEVICE}" \
  --eval-workers "${EVAL_WORKERS}" \
  --gpu-ids ${GPU_IDS_TEXT}
