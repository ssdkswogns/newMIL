#!/bin/bash

ROOT="./savemodel/millet_mixed_data"

# 다시 돌리고 싶은 데이터셋만 지정하면 아래 리스트를 사용.
# 비워두면 ROOT 아래 모든 폴더를 자동 수집.
RETRY_DATASETS=(
  "ArticularyWordRecognition"
  # "PhonemeSpectra"
  # "RacketSports"
  # "SelfRegulationSCP1"
  # "SelfRegulationSCP2"
  # "UWaveGestureLibrary"
  # "StandWalkJump"
)

# dataset 리스트 결정
if [[ ${#RETRY_DATASETS[@]} -gt 0 ]]; then
  DATASETS=("${RETRY_DATASETS[@]}")
else
  DATASETS=()
  if [[ -d "$ROOT" ]]; then
    while IFS= read -r d; do
      DATASETS+=("$(basename "$d")")
    done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
fi

# 반복 실행
for d in "${DATASETS[@]}"; do
  latest_exp_dir=$(ls -d "$ROOT/$d"/exp_* 2>/dev/null | sort -V | tail -n 1)

  if [[ -z "$latest_exp_dir" ]]; then
    echo "Skipping $d (no exp_* directory found)"
    continue
  fi

  model_path="$latest_exp_dir/weights/best_MILLET.pth"

  if [[ ! -f "$model_path" ]]; then
    echo "Skipping $d (missing model: $model_path)"
    continue
  fi

  echo "Evaluating $d using $model_path"

  python eval.py --dataset "$d" --model MILLET \
    --aopcr_stop 0.8 \
    --aopcr_step 0.05 \
    --model_path "$model_path" \
    --embed 128 --dropout_node 0.1 --millet_pooling conjunctive \
    --datatype mixed --batchsize 64 --cls_threshold 0.5 --num_random 3 --gpu_index 0 \
    --plot_aopcr \
    --aopcr_plot_path aopcr_curve.png \
    --seed 0
done
