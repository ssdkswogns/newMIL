#!/bin/bash
# Run OS_CNN on a list of datasets (original/single-label) using main_cl_fix_ambiguous.py

DATASETS=(
  # "ArticularyWordRecognition"
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
  # "Libras"
  # "LSST"
  # "NATOPS"
  # "PenDigits"
  # "PEMS-SF"
  # "PhonemeSpectra"
  # "RacketSports"
  # "SelfRegulationSCP1"
  # "SelfRegulationSCP2"
  # "StandWalkJump"
  # "UWaveGestureLibrary"
)

for dataset in "${DATASETS[@]}"; do
  echo "[OS_CNN] $dataset"
  CUDA_VISIBLE_DEVICES="" python3 main_cl_fix_ambiguous.py \
    --dataset "$dataset" \
    --datatype original \
    --model OS_CNN \
    --num_epochs 1500 \
    --batchsize 64 \
    --lr 5e-3 \
    --dropout_node 0.2 \
    --seed 0
  echo ""
done
