#!/bin/bash

# 실행할 데이터셋 목록
DATASETS=(
  "ArticularyWordRecognition"
  # # "AtrialFibrillation"
  # "BasicMotions"
  # "Cricket"
  # "DuckDuckGeese"
  # "Epilepsy"
  # "EthanolConcentration"
  # "ERing"
  # "FaceDetection"
  # "FingerMovements"
  # "HandMovementDirection"
  # "Handwriting"
  # # "Heartbeat"
  # "Libras"
  # "LSST"
  # "MotorImagery"
  # "NATOPS"
  # "PenDigits"
  # "PEMS-SF"
  # "PhonemeSpectra"
  # "RacketSports"
  # "SelfRegulationSCP1"
  # "SelfRegulationSCP2"
  # "StandWalkJump"
  # "UWaveGestureLibrary"
  # "JapaneseVowels"
)

# 반복 실행
for dataset in "${DATASETS[@]}"; do
    echo "Running TimeMIL on dataset: $dataset"
    python main_cl_fix_ambiguous.py  --dataset "$dataset" --model MILLET  --millet_pooling conjunctive   --embed 128  --dropout_node 0.1  --dropout_patch 0  --num_epochs 1500  --lr 5e-3 --datatype mixed
done