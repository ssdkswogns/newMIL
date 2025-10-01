#!/bin/bash

# 실행할 데이터셋 목록
DATASETS=(
  "BasicMotions"
  "Epilepsy"
  "Heartbeat"
  "PEMS-SF"
  "PhonemeSpectra"
  "StandWalkJump"
)

# 반복 실행
for dataset in "${DATASETS[@]}"; do
    echo "Running TimeMIL on dataset: $dataset"
    python main_exp.py --dataset "$dataset" --model newTimeMIL
done