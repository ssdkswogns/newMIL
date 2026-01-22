#!/bin/bash

DIST_DIR=./dist_mats/mixed
mkdir -p "$DIST_DIR"

# 실행할 데이터셋 목록
DATASETS=(
  "ArticularyWordRecognition"
  # "AtrialFibrillation"
  "BasicMotions"
  "Cricket"
  "DuckDuckGeese"
  "Epilepsy"
  # "EthanolConcentration"
  "ERing"
  "FaceDetection"
  "FingerMovements"
  "HandMovementDirection"
  "Handwriting"
  # "Heartbeat"
  "Libras"
  "LSST"
  "MotorImagery"
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
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29600 \
    main_cl_exp_softclt.py --dataset "$dataset" --model AmbiguousMIL --datatype mixed --use_softclt_aux --sim_mat_path "$DIST_DIR/soft_${dataset}_mixed_train.npz"  --epoch_des 20 --num_epochs 1500 --bag_loss_w 0.35 --inst_loss_w 0.35 --proto_loss_w 0.3
done