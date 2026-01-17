#!/bin/bash
set -euo pipefail

# Re-evaluate MILLET on DBA (bag metrics + instance acc for mixed).
ROOT="${ROOT:-./savemodel/InceptBackbone/DBA}"
EXP_DIR="${EXP_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"

DATATYPE="${DATATYPE:-mixed}"  # mixed | original
DROOT="${DROOT:-/home/giyong/newMIL/data/dba_parser}"
TEST_RATIO="${TEST_RATIO:-0.2}"
CONCAT_K="${CONCAT_K:-2}"

if [[ "${DATATYPE}" == "original" ]]; then
  WINDOW="${WINDOW:-30000}"
  STRIDE="${STRIDE:-15000}"
else
  WINDOW="${WINDOW:-12000}"
  STRIDE="${STRIDE:-6000}"
fi

MILLET_POOLING="${MILLET_POOLING:-conjunctive}"
EMBED="${EMBED:-128}"
DROPOUT_NODE="${DROPOUT_NODE:-0.1}"
BATCHSIZE="${BATCHSIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -z "${EXP_DIR}" ]]; then
    EXP_DIR="$(ls -d "${ROOT}"/exp_* 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  if [[ -z "${EXP_DIR}" ]]; then
    echo "No exp_* directory found under ${ROOT}"
    exit 1
  fi
  MODEL_PATH="${EXP_DIR}/weights/best_MILLET.pth"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Checkpoint not found: ${MODEL_PATH}"
  exit 1
fi

LOG_PATH="${LOG_PATH:-${MODEL_PATH%/*}/eval_dba.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" /home/giyong/miniconda3/envs/timemil/bin/python eval_dba.py \
  --model_path "${MODEL_PATH}" \
  --datatype "${DATATYPE}" \
  --dba_root "${DROOT}" \
  --dba_window "${WINDOW}" \
  --dba_stride "${STRIDE}" \
  --dba_test_ratio "${TEST_RATIO}" \
  --dba_concat_k "${CONCAT_K}" \
  --embed "${EMBED}" \
  --dropout_node "${DROPOUT_NODE}" \
  --millet_pooling "${MILLET_POOLING}" \
  --batchsize "${BATCHSIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --cls_threshold 0.5 | tee "${LOG_PATH}"
