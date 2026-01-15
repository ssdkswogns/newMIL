#!/bin/bash
# Reproduce exp_65 settings (AmbiguousMIL, no loss schedule) for selected datasets.

set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYBIN="/home/moon/anaconda3/envs/timemil/bin/python"
DATASETS=(
  "ArticularyWordRecognition"
  # "BasicMotions"
  # "Cricket"
  # "DuckDuckGeese"
  # "Epilepsy"
  # # "EthanolConcentration"
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
for DATASET in "${DATASETS[@]}"; do
  SAVE_DIR="./savemodel/InceptBackbone/${DATASET}/exp_65_repro_drop0.1"
  echo "=== Repro (AmbiguousMIL, no schedule) on ${DATASET} ==="
  "${PYBIN}" main_cl_fix_ambiguous.py \
    --dataset "${DATASET}" \
    --datatype mixed \
    --model AmbiguousMIL \
    --embed 128 \
    --dropout_node 0.1 \
    --dropout_patch 0.5 \
    --lr 3e-3 \
    --num_epochs 1500 \
    --epoch_des 20 \
    --device 1 \
    --bag_loss_w 0.35 \
    --inst_loss_w 0.35 \
    --proto_loss_w 0.3 \
    --proto_tau 0.2 \
    --proto_sim_thresh 0.5 \
    --proto_win 30 \
    --ctx_tau 0.1 \
    --ctx_win 4 \
    --sparsity_loss_w 0.0 \
    --optimizer adamw \
    --weight_decay 1e-4 \
    --seed 0 \
    --num_workers 0 \
    --gpu_index 0 \
    --save_dir "${SAVE_DIR}"
  echo
  sleep 2
done
