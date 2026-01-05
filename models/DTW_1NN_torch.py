"""
Torch-based DTW 1-NN baselines (CPU/GPU). Meant for small/medium sequences as a
drop-in replacement for the dtaidistance/tslearn versions, with two flavors:
  - Independent (DTW1NN_I): per-channel DTW summed across channels
  - Dependent   (DTW1NN_D): multivariate DTW with shared warping path
Inputs X_* are numpy arrays shaped (N, T, C). Outputs are numpy label arrays.
"""
from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def _dtw_distance_1d(x: torch.Tensor, y: torch.Tensor, window: int = None) -> torch.Tensor:
    """DTW distance for 1D sequences x, y (shape [T]); optional Sakoe-Chiba window."""
    n, m = x.shape[0], y.shape[0]
    cost = torch.abs(x.view(n, 1) - y.view(1, m))
    dp = torch.full((n + 1, m + 1), float("inf"), device=x.device)
    dp[0, 0] = 0.0
    for i in range(n):
        j_min = 0
        j_max = m
        if window is not None:
            j_min = max(0, i - window)
            j_max = min(m, i + window + 1)
        for j in range(j_min, j_max):
            dp[i + 1, j + 1] = cost[i, j] + torch.min(
                torch.stack([dp[i, j + 1], dp[i + 1, j], dp[i, j]])
            )
    return dp[n, m]


@torch.no_grad()
def _dtw_distance_dependent(x: torch.Tensor, y: torch.Tensor, window: int = None) -> torch.Tensor:
    """
    Multivariate DTW with shared warping path.
    x, y: [T, C]
    """
    n, m = x.shape[0], y.shape[0]
    cost = torch.cdist(x, y)  # [n, m] Euclidean per timestep
    dp = torch.full((n + 1, m + 1), float("inf"), device=x.device)
    dp[0, 0] = 0.0
    for i in range(n):
        j_min = 0
        j_max = m
        if window is not None:
            j_min = max(0, i - window)
            j_max = min(m, i + window + 1)
        for j in range(j_min, j_max):
            dp[i + 1, j + 1] = cost[i, j] + torch.min(
                torch.stack([dp[i, j + 1], dp[i + 1, j], dp[i, j]])
            )
    return dp[n, m]


@torch.no_grad()
def dtw_1nn_I_predict_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: Optional[torch.device] = None,
    window: int = None,
) -> np.ndarray:
    """
    Independent-channel DTW 1-NN.
    X_*: (N, T, C); per-channel DTW distance summed across channels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = np.asarray(y_train)

    preds = []
    for x in Xte:  # x: [T, C]
        best_d = float("inf")
        best_y = None
        for xt, yt in zip(Xtr, y_train):
            d = 0.0
            for c in range(x.shape[1]):
                d = d + _dtw_distance_1d(x[:, c], xt[:, c], window=window).item()
            if d < best_d:
                best_d = d
                best_y = yt
        preds.append(best_y)
    return np.array(preds)


@torch.no_grad()
def dtw_1nn_D_predict_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: Optional[torch.device] = None,
    window: int = None,
) -> np.ndarray:
    """
    Dependent multivariate DTW 1-NN (shared path).
    X_*: (N, T, C)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = np.asarray(y_train)

    preds = []
    for x in Xte:  # [T, C]
        best_d = float("inf")
        best_y = None
        for xt, yt in zip(Xtr, y_train):
            d = _dtw_distance_dependent(x, xt, window=window).item()
            if d < best_d:
                best_d = d
                best_y = yt
        preds.append(best_y)
    return np.array(preds)
