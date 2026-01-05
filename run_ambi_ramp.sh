#!/bin/bash
# Run AmbiguousMIL with CL ramp (ramp engine) using the same hparams as run_ambi_cl.sh for fair comparison.

DATASETS=(
  "ArticularyWordRecognition"
  "BasicMotions"
  "Cricket"
  "DuckDuckGeese"
  "Epilepsy"
)

for d in "${DATASETS[@]}"; do
  echo "Running AmbiguousMIL + CL (ramp) on dataset: $d"
  python3 main_cl_fix_ramp.py \
    --dataset "$d" \
    --datatype mixed \
    --model AmbiguousMIL \
    --epoch_des 40 \
    --num_epochs 1500 \
    --bag_loss_w 0.3 \
    --inst_loss_w 0.3 \
    --proto_loss_w 0.4 \
    --lr 5e-3 \
    --proto_ramp_start 50 \
    --proto_ramp_len 30 \
    --batch_size 256

  echo "Done: $d"
  echo
  sleep 2
done
