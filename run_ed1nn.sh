#!/bin/bash
# Baseline sweep using ED1NN for original (single-label) datasets, matching datasets in run_milet.sh

DATASETS=(
  "ArticularyWordRecognition"
  # "AtrialFibrillation"  # ED1NN supports original only; keep list aligned
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

for dataset in "${DATASETS[@]}"; do
  echo "[ED1NN] Running on dataset: $dataset"
  python3 main_cl_fix_ambiguous.py \
    --dataset "$dataset" \
    --datatype original \
    --model ED1NN \
    # --ed1nn_normalize
  echo "" 
done
