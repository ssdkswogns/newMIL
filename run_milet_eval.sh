#!/bin/bash

# 실행할 데이터셋 목록
DATASETS=(
  "ArticularyWordRecognition"
  # "AtrialFibrillation"
  "BasicMotions"
  "Cricket"
  "DuckDuckGeese"
  "Epilepsy"
  "EthanolConcentration"
  "ERing"
  "FaceDetection"
  "FingerMovements"
  "HandMovementDirection"
  "Handwriting"
  # "Heartbeat"
  "Libras"
  "LSST"
  # "MotorImagery"
  "NATOPS"
  "PenDigits"
  "PEMS-SF"
  "PhonemeSpectra"
  "RacketSports"
  "SelfRegulationSCP1"
  "SelfRegulationSCP2"
  "StandWalkJump"
  "UWaveGestureLibrary"
  # "JapaneseVowels"
)

# 반복 실행
for d in "${DATASETS[@]}"; do
  latest_exp_dir=$(ls -d "./savemodel/InceptBackbone/$d"/exp_* 2>/dev/null | sort -V | tail -n 1)

  if [[ -z "$latest_exp_dir" ]]; then
    echo "Skipping $d (no exp_* directory found)"
    continue
  fi

  model_path="$latest_exp_dir/weights/best_MILLET.pth"

  if [[ ! -f "$model_path" ]]; then
    echo "Skipping $d (missing model: $model_path)"
    continue
  fi

  echo "Evaluating $d using $model_path"

  python eval.py --dataset "$d" --model MILLET \
    --model_path "$model_path" \
    --embed 128 --dropout_node 0.1 --millet_pooling conjunctive \
    --datatype mixed --batchsize 64 --cls_threshold 0.5 --num_random 3 --gpu_index 0
done
