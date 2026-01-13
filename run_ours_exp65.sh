#!/bin/bash
# Reproduce exp_65 settings (ours) on ArticularyWordRecognition

python main_cl_fix_ambiguous.py \
  --dataset ArticularyWordRecognition \
  --datatype mixed \
  --model AmbiguousMIL \
  --num_classes 2 \
  --num_epochs 1500 \
  --epoch_des 20 \
  --batchsize 64 \
  --lr 0.005 \
  --weight_decay 0.0001 \
  --optimizer adamw \
  --dropout_node 0.2 \
  --dropout_patch 0.5 \
  --embed 128 \
  --ctx_win 4 \
  --ctx_tau 0.1 \
  --ctx_contrast_w 0.0 \
  --cls_contrast_tau 0.1 \
  --cls_contrast_w 0.0 \
  --bag_loss_w 0.35 \
  --inst_loss_w 0.35 \
  --proto_loss_w 0.3 \
  --proto_tau 0.2 \
  --proto_sim_thresh 0.5 \
  --proto_win 30 \
  --smooth_loss_w 0.05 \
  --sparsity_loss_w 0.05 \
  --num_workers 0 \
  --seed 0 \
  --feats_size 512 \
  --save_dir ./savemodel/
