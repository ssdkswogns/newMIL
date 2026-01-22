#!/bin/bash

LOSSES=(
  0.25
  0.3
  0.35
  0.4
  0.45
)

for loss in "${LOSSES[@]}"; do
    # proto_loss_w = 1 - 2 * loss
    proto_loss_w=$(echo "1 - 2 * $loss" | bc -l)

    echo "Running TimeMIL on dataset: JapaneseVowels"
    echo "bag_loss_w = $loss, inst_loss_w = $loss, proto_loss_w = $proto_loss_w"

    CUDA_VISIBLE_DEVICES=2,3 torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        main_cl_exp.py \
        --dataset JapaneseVowels \
        --model AmbiguousMIL \
        --datatype mixed \
        --bag_loss_w $loss \
        --inst_loss_w $loss \
        --proto_loss_w $proto_loss_w \
        --epoch_des 20 \
        --num_epochs 1500 \
        --proto_win 30 \
        --proto_tau 0.2
done