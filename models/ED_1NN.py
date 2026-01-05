#!/usr/bin/env python3
"""
Lightweight Euclidean Distance 1-NN classifier with a tiny CLI so it can be
run on aeon datasets (or a built-in toy demo).
"""
import argparse
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

try:
    # Optional; only needed when using the CLI with --dataset.
    from aeon.datasets import load_classification
except Exception:  # pragma: no cover - aeon is optional for the CLI
    load_classification = None


class ED1NN:
    """
    Euclidean Distance 1-NN classifier.
    Supports X shaped (n_samples, n_features) or (n_samples, T, C) by flattening.
    """

    def __init__(self, normalize: bool = False):
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self._mean = None
        self._std = None

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X.reshape(X.shape[0], -1).astype(np.float32)

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X_flat = self._flatten(X)
        if not self.normalize:
            return X_flat

        if fit:
            self._mean = X_flat.mean(axis=0, keepdims=True)
            self._std = X_flat.std(axis=0, keepdims=True) + 1e-8

        if self._mean is None or self._std is None:
            raise ValueError("Model must be fit before calling predict when normalization is enabled.")

        return (X_flat - self._mean) / self._std

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train = self._transform(X, fit=True)
        self.y_train = y
        return self

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call fit before predict.")

        X_proc = self._transform(X, fit=False)

        # (n_test, n_train) 거리 행렬: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        X2 = np.sum(X_proc * X_proc, axis=1, keepdims=True)              # (n_test, 1)
        T2 = np.sum(self.X_train * self.X_train, axis=1)[None]           # (1, n_train)
        d2 = X2 + T2 - 2.0 * (X_proc @ self.X_train.T)                   # (n_test, n_train)
        nn_idx = np.argmin(d2, axis=1)
        return self.y_train[nn_idx]

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))


def _load_aeon_dataset(name: str, extract_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if load_classification is None:
        raise RuntimeError("aeon is not installed; install it or run the --demo path.")

    Xtr, ytr, _ = load_classification(name=name, split="train", extract_path=extract_path, return_metadata=True)
    Xte, yte, _ = load_classification(name=name, split="test", extract_path=extract_path, return_metadata=True)
    return Xtr, ytr, Xte, yte


def run_demo(normalize: bool) -> None:
    X_train = np.array([[0, 0], [1, 1], [10, 10]])
    y_train = np.array([0, 0, 1])
    X_test = np.array([[0.2, 0.1], [9, 9]])

    clf = ED1NN(normalize=normalize).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(np.array([0, 1]), y_pred)
    print("Demo complete")
    print(f"Predictions: {y_pred}")
    print(f"Accuracy:    {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run the ED_1NN baseline on an aeon dataset or a tiny demo.")
    parser.add_argument("--dataset", type=str, default=None, help="aeon dataset name (e.g., BasicMotions, PenDigits)")
    parser.add_argument("--extract-path", type=str, default="./data", help="Path where aeon stores/reads datasets")
    parser.add_argument("--normalize", action="store_true", help="Z-score features using training mean/std before 1-NN")
    parser.add_argument("--demo", action="store_true", help="Run a small built-in sanity check instead of a dataset")
    args = parser.parse_args()

    if args.demo or args.dataset is None:
        run_demo(args.normalize)
        return

    Xtr, ytr, Xte, yte = _load_aeon_dataset(args.dataset, args.extract_path)
    clf = ED1NN(normalize=args.normalize).fit(Xtr, ytr)
    y_pred = clf.predict(Xte)

    acc = accuracy_score(yte, y_pred)
    bal_acc = balanced_accuracy_score(yte, y_pred)
    f1_ma = f1_score(yte, y_pred, average="macro")

    print(f"Dataset: {args.dataset}")
    print(f"Train: {Xtr.shape} | Test: {Xte.shape}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"F1 (macro):        {f1_ma:.4f}")


if __name__ == "__main__":
    main()
