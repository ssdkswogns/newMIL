#!/bin/bash

DATASETS=(
  "ArticularyWordRecognition"
  "BasicMotions"
  "Cricket"
  "DuckDuckGeese"
  "Epilepsy"
  "ERing"
  "FingerMovements"
  "HandMovementDirection"
  "NATOPS"
  "SelfRegulationSCP1"
  "StandWalkJump"
  "UWaveGestureLibrary"
)

BASE_DIR=./savemodel/InceptBackbone/Ambi_UEA_SOTA

for dataset in "${DATASETS[@]}"; do
    echo "Running Visualize on dataset: $dataset"

    # exp_* 디렉토리 탐색 (하나만 있다고 가정)
    EXP_DIR=$(ls -d ${BASE_DIR}/${dataset}/exp_* 2>/dev/null)

    if [ -z "$EXP_DIR" ]; then
        echo "[Warning] No exp_* directory found for $dataset"
        continue
    fi

    CKPT_PATH=${EXP_DIR}/weights/best_AmbiguousMIL.pth

    if [ ! -f "$CKPT_PATH" ]; then
        echo "[Warning] Checkpoint not found: $CKPT_PATH"
        continue
    fi

    python visualize.py \
        --dataset "$dataset" \
        --model AmbiguousMIL \
        --datatype mixed \
        --ckpt "$CKPT_PATH" \
        --save_dir ./explain_pred_ambi \
        --batch_size 128 \
        --max_save 200
done