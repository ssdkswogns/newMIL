#!/bin/bash
# Evaluate exp_65 (ours) checkpoint on ArticularyWordRecognition

CKPT=./savemodel/ours/exp_65/weights/best_AmbiguousMIL.pth

python eval.py \
  --dataset ArticularyWordRecognition \
  --datatype mixed \
  --model HybridMIL \
  --hybrid_bag_head tp \
  --model_path "${CKPT}" \
  --embed 128 \
  --feats_size 512 \
  --dropout_node 0.2 \
  --batchsize 1 \
  --num_workers 0 \
  --cls_threshold 0.5 \
  --num_random 3 \
  --aopcr_stop 0.8 \
  --aopcr_step 0.05 \
  --plot_aopcr \
  --aopcr_plot_path aopcr_curve.png \
  --seed 0
