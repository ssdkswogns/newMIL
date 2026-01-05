#!/bin/bash
# Baseline sweep using DTW-1NN (independent and dependent variants) on original (single-label) datasets.
# Uses the torch implementations (GPU if available).

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

MODELS=("DTW1NN_I" "DTW1NN_D")

for model in "${MODELS[@]}"; do
  echo "===== $model sweep ====="
  for dataset in "${DATASETS[@]}"; do
    echo "[$model] Running on dataset: $dataset"
    # 창(window) 제한과 train/test 샘플 제한으로 속도 개선
    python3 main_cl_fix_ambiguous.py \
      --dataset "$dataset" \
      --datatype original \
      --model "$model" \
      --dtw_window 20 \
      --dtw_max_train 200 \
      --dtw_max_test 200
    echo ""
  done
done
