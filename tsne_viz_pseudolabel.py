# -*- coding: utf-8 -*-
"""
Reuse saved t-SNE core (XY + flat_idx) and recolor by pseudo labels.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = {
    0: "Aggressive",
    1: "Conservative",
    2: "Normal",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsne_core", type=str, required=True)
    parser.add_argument("--pseudo_npz", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./tsne_out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load tsne core
    core = np.load(args.tsne_core)
    XY = core["XY"]
    flat_idx = core["flat_idx"]
    stride = int(core["stride_t"])

    # load pseudo labels
    d = np.load(args.pseudo_npz)
    Y_hat = d["Y_hat"]              # [N,T]

    # flatten pseudo labels with SAME stride
    Ys = Y_hat[:, ::stride]
    y_flat = Ys.reshape(-1)

    y = y_flat[flat_idx]            # ★ 핵심 정합

    # plot
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

    plt.title("t-SNE of AE per-timestep features (colored by pseudo labels)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Predicted driving style", markerscale=3, frameon=True)

    out_png = os.path.join(args.out_dir, "tsne_pseudolabel.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[save] {out_png}")


if __name__ == "__main__":
    main()
