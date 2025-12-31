#!/bin/bash
# Run AmbiguousMIL with contrastive proto loss (cl_w=0.4) and equal bag/instance weights.

DATASETS=(
  "ArticularyWordRecognition"
  "BasicMotions"
  "Cricket"
  "DuckDuckGeese"
  "Epilepsy"
)

for d in "${DATASETS[@]}"; do
  echo "Running AmbiguousMIL + CL on dataset: $d"
  python3 main_cl_fix_ambiguous.py \
    --dataset "$d" \
    --datatype mixed \
    --model AmbiguousMIL \
    --epoch_des 20 \
    --num_epochs 1500 \
    --bag_loss_w 0.3 \
    --inst_loss_w 0.3 \
    --proto_loss_w 0.4
    
  echo "Done: $d"
  echo
  sleep 2
done
