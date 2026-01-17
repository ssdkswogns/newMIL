#!/bin/bash
set -euo pipefail

# MILLET training on DBA dataset
DATASET="DBA"
DATATYPE="mixed"           # options: mixed | original

DROOT="/home/moon/code/newMIL/data/dba_parser"
WINDOW=12000
STRIDE=6000
TEST_RATIO=0.2
CONCAT_K=2                 # used only when DATATYPE=mixed

MODEL="MILLET"
MILLET_POOLING="conjunctive"

python3 main_cl_fix_ambiguous.py \
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
  --batchsize 4
