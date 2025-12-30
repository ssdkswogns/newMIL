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
# for dataset in "${DATASETS[@]}"; do
#     echo "Running TimeMIL on dataset: $dataset"
#     python main.py --dataset "$dataset" --model TimeMIL
# done

for dataset in "${DATASETS[@]}"; do
    echo "Running TimeMIL on dataset: $dataset"
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29500 main_cl_fix.py --dataset $dataset --model AmbiguousMIL --datatype mixed --bag_loss_w 0.15 --inst_loss_w 0.35 --proto_loss_w 0.5 --epoch_des 20 --num_epochs 1500
done