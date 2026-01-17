#!/bin/bash
set -euo pipefail

# MILLET training on DBA dataset
DATASET="DBA"
DATATYPE="original"           # options: mixed | original

DROOT="${DROOT:-/home/giyong/newMIL/data/dba_parser}"
# original 추천: WINDOW=30000, STRIDE=15000
# mixed 추천:    WINDOW=12000, STRIDE=6000
WINDOW=30000
STRIDE=15000
TEST_RATIO=0.2
CONCAT_K=2                 # used only when DATATYPE=mixed

MODEL="MILLET"
MILLET_POOLING="conjunctive"

# Device/DDP controls
# - Single GPU: export CUDA_VISIBLE_DEVICES=0 (or set below)
# - CPU only:  export CUDA_VISIBLE_DEVICES=""
# - DDP:       export DDP_NPROC=2 (uses torchrun)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
DDP_NPROC="${DDP_NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ "${DDP_NPROC}" -gt 1 ]]; then
  MASTER_PORT="${MASTER_PORT:-29500}"
  LAUNCHER=(torchrun --nproc_per_node="${DDP_NPROC}" --rdzv_backend=c10d --rdzv_endpoint="localhost:${MASTER_PORT}")
else
  LAUNCHER=(python3)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${LAUNCHER[@]}" main_cl_fix_ambiguous.py \
  --dataset "${DATASET}" \
  --datatype "${DATATYPE}" \
  --model "${MODEL}" \
  --millet_pooling "${MILLET_POOLING}" \
  --dba_root "${DROOT}" \
  --dba_window "${WINDOW}" \
  --dba_stride "${STRIDE}" \
  --dba_test_ratio "${TEST_RATIO}" \
  --dba_concat_k "${CONCAT_K}" \
  --embed 128 \
  --dropout_node 0.1 \
  --dropout_patch 0 \
  --num_epochs 1500 \
  --lr 5e-3 \
  --batchsize 64
