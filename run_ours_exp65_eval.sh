#!/bin/bash
# Evaluate exp_65 (ours) checkpoint on ArticularyWordRecognition

# CKPT=./savemodel/ours/exp_65/weights/best_AmbiguousMIL.pth
# --class_order meta

CKPT=./savemodel/InceptBackbone/ArticularyWordRecognition/exp_65_reproInceptBackbone/ArticularyWordRecognition/exp_10/weights/best_AmbiguousMIL.pth
/home/moon/anaconda3/envs/timemil/bin/python eval.py \
  --dataset ArticularyWordRecognition \
  --datatype mixed \
  --model AmbiguousMIL \
  --batchsize 64 \
  --dropout_node 0.2 \
  --model_path "${CKPT}" \
  --embed 128 \
  --feats_size 512 \
  --dropout_node 0.2 \
  --num_workers 0 \
  --cls_threshold 0.5 \
  --num_random 3 \
  --aopcr_stop 0.5 \
  --aopcr_step 0.05 \
  --plot_aopcr \
  --aopcr_plot_path aopcr_curve.png \
  --seed 0  
  # --class_order meta