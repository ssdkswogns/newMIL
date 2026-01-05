#!/bin/bash
# Run AEON's WEASEL+MUSE classifier on a list of datasets (original/single-label).
# Requires: pip install aeon

DATASETS=(
  "ArticularyWordRecognition"
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
  "Libras"
  "LSST"
  "NATOPS"
  "PenDigits"
  "PEMS-SF"
  "PhonemeSpectra"
  "RacketSports"
  "SelfRegulationSCP1"
  "SelfRegulationSCP2"
  "StandWalkJump"
  "UWaveGestureLibrary"
)

for dataset in "${DATASETS[@]}"; do
  echo "[MUSE] $dataset"
  AEON_DATA_DIR="${AEON_DATA_DIR:-./data}"
  DATASET_NAME="$dataset" AEON_DATA_DIR="$AEON_DATA_DIR" timeout 60s python3 - <<'PY'
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from aeon.datasets import load_classification
from aeon.classification.dictionary_based import MUSE
import os
import sys
import urllib.error

name = os.environ.get("DATASET_NAME")
data_dir = os.environ.get("AEON_DATA_DIR", "./data")
try:
    Xtr, ytr = load_classification(name=name, split="train", extract_path=data_dir, return_metadata=False)
    Xte, yte = load_classification(name=name, split="test", extract_path=data_dir, return_metadata=False)
except (urllib.error.HTTPError, urllib.error.URLError) as e:
    print(f"[MUSE] download failed for {name}: {e}")
    sys.exit(1)

clf = MUSE(n_jobs=-1)
clf.fit(Xtr, ytr)
y_pred = clf.predict(Xte)

acc = accuracy_score(yte, y_pred)
bal = balanced_accuracy_score(yte, y_pred)
f1m = f1_score(yte, y_pred, average="macro")
print(f"Acc={acc:.4f} BalAcc={bal:.4f} F1m={f1m:.4f}")
PY
  if [ $? -ne 0 ]; then
    echo "[MUSE] $dataset skipped (download/error). Pre-download dataset into \$AEON_DATA_DIR to run."
  fi
done
