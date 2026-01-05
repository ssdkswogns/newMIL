#!/bin/bash
# Train MLSTM_FCN (PyTorch) on a list of aeon datasets (single-label).

DATASETS=(
  "ArticularyWordRecognition"
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
  "Libras"
  "LSST"
  "NATOPS"
  "PenDigits"
  "PEMS-SF"
  "PhonemeSpectra"
  "RacketSports"
  "SelfRegulationSCP1"
  "SelfRegulationSCP2"
  "StandWalkJump"
  "UWaveGestureLibrary"
)

for dataset in "${DATASETS[@]}"; do
  echo "[MLSTM_FCN] $dataset"
  # main_cl_fix_ambiguous 파이프라인에서 MLSTM_FCN 실행 (AOPCR 연동 가능)
  # CUDA 불안 시 CUDA_VISIBLE_DEVICES="" 와 --gpu_index 비우기(또는 cpu 강제)
  CUDA_VISIBLE_DEVICES="" python3 main_cl_fix_ambiguous.py \
    --dataset "$dataset" \
    --datatype mixed \
    --model MLSTM_FCN \
    --num_epochs 1500 \
    --gpu_index 0
  echo ""
done
