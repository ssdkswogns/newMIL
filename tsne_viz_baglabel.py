# -*- coding: utf-8 -*-
"""
Build t-SNE ONCE from AE embeddings and save:
  - XY coordinates
  - flat indices (which timesteps were used)

Then plot with BAG-level labels.

Later, the SAME saved XY + flat_idx can be reused
to plot pseudo-labels, confidence, attention, etc.

Usage:
python tsne_build_and_plot_baglabel.py \
  --emb_npz ./savemodel_ae/dba_ae/dba_ae_embeddings.npz \
  --label_npy ./savemodel_ae/dba_ae/y_bag.npy \
  --out_dir ./tsne_out \
  --stride_t 5 \
  --max_points 200000 \
  --perplexity 30 \
  --seed 42
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# utils
# -------------------------------------------------
def maybe_mkdir_p(p: str):
    os.makedirs(p, exist_ok=True)


def load_labels(path: str):
    if path.endswith(".npy"):
        y = np.load(path)
    elif path.endswith(".npz"):
        d = np.load(path)
        for k in ["y", "Y", "labels", "label"]:
            if k in d:
                y = d[k]
                break
        else:
            raise KeyError(f"Cannot find label key in {path}")
    else:
        raise ValueError("label file must be .npy or .npz")

    y = np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(axis=1)
    return y.astype(np.int64)


# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Build t-SNE once and plot bag-label view")
    parser.add_argument("--emb_npz", type=str, required=True)
    parser.add_argument("--label_npy", type=str, required=True)

    parser.add_argument("--out_dir", type=str, default="./tsne_out")
    parser.add_argument("--stride_t", type=int, default=5)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning_rate", type=float, default=200.0)
    parser.add_argument("--n_iter", type=int, default=1000)

    parser.add_argument("--standardize", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    maybe_mkdir_p(args.out_dir)

    CLASS_NAMES = {
        0: "Aggressive",
        1: "Conservative",
        2: "Normal",
    }

    # -------------------------------------------------
    # 1) load embeddings
    # -------------------------------------------------
    emb = np.load(args.emb_npz)
    Z = emb["Z"]                    # [N,T,D]
    N, T, D = Z.shape
    print(f"[load] Z={Z.shape}")

    # -------------------------------------------------
    # 2) load bag labels
    # -------------------------------------------------
    y_bag = load_labels(args.label_npy)
    assert len(y_bag) == N
    print(f"[load] y_bag={y_bag.shape}")

    # -------------------------------------------------
    # 3) flatten with stride
    # -------------------------------------------------
    stride = max(1, args.stride_t)
    Zs = Z[:, ::stride, :]           # [N,T',D]
    Tp = Zs.shape[1]

    X_all = Zs.reshape(N * Tp, D)    # [N*Tp,D]
    y_all = np.repeat(y_bag, Tp)     # [N*Tp]

    print(f"[prep] stride={stride} -> X_all={X_all.shape}")

    # -------------------------------------------------
    # 4) subsample (SAVE INDICES!)
    # -------------------------------------------------
    if args.max_points > 0 and X_all.shape[0] > args.max_points:
        flat_idx = np.random.choice(
            X_all.shape[0],
            size=args.max_points,
            replace=False
        )
    else:
        flat_idx = np.arange(X_all.shape[0])

    X = X_all[flat_idx]
    y = y_all[flat_idx]

    print(f"[subsample] X={X.shape}")

    # -------------------------------------------------
    # 5) preprocessing
    # -------------------------------------------------
    if args.standardize:
        X = StandardScaler().fit_transform(X)

    # -------------------------------------------------
    # 6) t-SNE (RUN ONCE)
    # -------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        init="pca",
        random_state=args.seed,
        verbose=1,
    )
    XY = tsne.fit_transform(X)
    print(f"[tsne] XY={XY.shape}")

    # -------------------------------------------------
    # 7) SAVE CORE ARTIFACT (VERY IMPORTANT)
    # -------------------------------------------------
    core_path = os.path.join(args.out_dir, "tsne_core.npz")
    np.savez_compressed(
        core_path,
        XY=XY,
        flat_idx=flat_idx,
        stride_t=stride,
    )
    print(f"[save] core -> {core_path}")

    # -------------------------------------------------
    # 8) plot: BAG LABEL
    # -------------------------------------------------
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for c in [0, 1, 2]:
        mask = (y == c)
        plt.scatter(
            XY[mask, 0],
            XY[mask, 1],
            s=3,
            alpha=0.45,
            color=cmap(c),
            label=CLASS_NAMES[c],
            linewidths=0,
        )

    plt.title("t-SNE of AE per-timestep features (colored by bag label)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Driving style", markerscale=3, frameon=True)

    out_png = os.path.join(args.out_dir, "tsne_baglabel.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[save] plot -> {out_png}")


if __name__ == "__main__":
    main()