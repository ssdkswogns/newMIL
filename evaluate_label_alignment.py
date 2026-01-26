# -*- coding: utf-8 -*-
"""
Fair label-feature alignment evaluation in FEATURE SPACE (NOT t-SNE space).

- Compute clustering metrics on AE features (Z), using the SAME sampled points (flat_idx).
- t-SNE is only for visualization; metrics are computed on feature vectors.

Metrics (computed on features X_feat):
  - Silhouette Score (↑)
  - Davies–Bouldin Index (↓)
  - Calinski–Harabasz Index (↑)

Required:
  1) AE embeddings npz: key 'Z' with shape [N, T, z_dim]
  2) tsne_core npz: contains 'flat_idx' and 'stride_t' (from your t-SNE build script)
     - We reuse flat_idx to ensure we evaluate the SAME points as visualization.
  3) bag labels: y_bag.npy or npz ([N] or [N,C])
  4) pseudo labels npz: key 'Y_hat' [N, T] and confidence key in CONF_KEYS [N, T]
     - Y_hat must be hard class ids {0,1,2}
     - conf is max-prob (or any confidence scalar per timestep)

Usage:
python evaluate_label_alignment_in_feature_space.py \
  --emb_npz ./savemodel_ae/dba_ae/dba_ae_embeddings.npz \
  --tsne_core ./tsne_out/tsne_core.npz \
  --bag_label_npy ./savemodel_ae/dba_ae/y_bag.npy \
  --pseudo_npz ./tsne_out/pseudo_inst_labels.npz \
  --tau_list 0.0 0.5 0.7 0.8 0.9 \
  --standardize \
  --pca_dim 50
"""

import argparse
import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


CONF_KEYS = ["conf", "confidence", "Y_conf", "prob", "pmax", "max_prob"]


def load_labels(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        y = np.load(path)
    elif path.endswith(".npz"):
        d = np.load(path)
        for k in ["y", "Y", "labels", "label"]:
            if k in d:
                y = d[k]
                break
        else:
            raise KeyError(f"Cannot find label key in {path}. keys={list(d.keys())}")
    else:
        raise ValueError("label file must be .npy or .npz")

    y = np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(axis=1)
    return y.astype(np.int64)


def _load_conf(npz_obj) -> np.ndarray:
    for k in CONF_KEYS:
        if k in npz_obj:
            return npz_obj[k]
    raise KeyError(f"Cannot find confidence key in pseudo_npz. Tried={CONF_KEYS}, keys={list(npz_obj.keys())}")


def _safe_metrics(X, labels):
    """
    Compute metrics safely.
    - Require >=2 classes
    - Silhouette requires at least 2 samples per class to be meaningful
    """
    labels = labels.astype(np.int64)
    uniq, cnt = np.unique(labels, return_counts=True)
    out = {"sil": None, "db": None, "ch": None, "n": int(len(labels)), "k": int(len(uniq))}

    if out["n"] < 10 or out["k"] < 2:
        return out

    # DB/CH can work even if some class has 1 sample (but can still be unstable)
    try:
        out["db"] = float(davies_bouldin_score(X, labels))
    except Exception:
        out["db"] = None
    try:
        out["ch"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        out["ch"] = None

    if np.any(cnt < 2):
        # Silhouette unreliable if any class count < 2
        return out

    try:
        out["sil"] = float(silhouette_score(X, labels))
    except Exception:
        out["sil"] = None

    return out


def main():
    parser = argparse.ArgumentParser("Evaluate label alignment in FEATURE SPACE (fair)")
    parser.add_argument("--emb_npz", type=str, required=True, help="npz with Z=[N,T,z_dim]")
    parser.add_argument("--tsne_core", type=str, required=True, help="tsne_core.npz with flat_idx + stride_t")
    parser.add_argument("--bag_label_npy", type=str, required=True, help="bag labels y_bag.npy/npz ([N] or [N,C])")
    parser.add_argument("--pseudo_npz", type=str, required=True, help="pseudo labels npz with Y_hat [N,T] and conf [N,T]")
    parser.add_argument("--tau_list", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.8, 0.9])

    # feature preprocessing
    parser.add_argument("--standardize", action="store_true", help="Standardize features before metrics")
    parser.add_argument("--pca_dim", type=int, default=0,
                        help="If >0, apply PCA to this dimension before metrics (recommended for speed/stability)")

    args = parser.parse_args()

    # ----------------------------
    # Load embeddings (feature space)
    # ----------------------------
    emb = np.load(args.emb_npz)
    if "Z" not in emb:
        raise KeyError(f"emb_npz must contain key 'Z'. keys={list(emb.keys())}")
    Z = emb["Z"]  # [N,T,z_dim]
    N, Tz, Dz = Z.shape
    print(f"[load] Z={Z.shape}")

    # ----------------------------
    # Load tsne_core indices (point selection)
    # ----------------------------
    core = np.load(args.tsne_core)
    if "flat_idx" not in core or "stride_t" not in core:
        raise KeyError(f"tsne_core must contain flat_idx and stride_t. keys={list(core.keys())}")
    flat_idx = core["flat_idx"].astype(np.int64)
    stride = int(core["stride_t"])
    print(f"[load] tsne_core stride_t={stride}, flat_idx={flat_idx.shape}, max_idx={flat_idx.max()}")

    # ----------------------------
    # Build feature matrix for SAME points
    #   - Apply same stride along time
    #   - Flatten to [N*Tp, Dz]
    #   - Index by flat_idx to get [M, Dz]
    # ----------------------------
    Zs = Z[:, ::stride, :]                 # [N,Tp,Dz]
    Tp = Zs.shape[1]
    X_feat_all = Zs.reshape(N * Tp, Dz)    # [N*Tp, Dz]

    if flat_idx.max() >= X_feat_all.shape[0]:
        raise IndexError(
            f"flat_idx out of range for features: max(flat_idx)={flat_idx.max()} vs X_feat_all={X_feat_all.shape}. "
            f"(Possible mismatch: tsne_core built with different embeddings/stride/datatype.)"
        )

    X_feat = X_feat_all[flat_idx]          # [M, Dz]
    M = X_feat.shape[0]
    print(f"[prep] X_feat(all)={X_feat_all.shape} -> X_feat(used)={X_feat.shape}")

    # ----------------------------
    # Load bag labels -> per-timestep (after stride) -> align with flat_idx
    # ----------------------------
    y_bag = load_labels(args.bag_label_npy)   # [N]
    if len(y_bag) != N:
        raise ValueError(f"bag label N mismatch: y_bag={len(y_bag)} vs Z N={N}")

    y_bag_flat = np.repeat(y_bag, Tp)         # [N*Tp]
    y_bag_used = y_bag_flat[flat_idx]         # [M]

    # ----------------------------
    # Load pseudo labels -> stride -> flatten -> align with flat_idx
    # ----------------------------
    pseudo = np.load(args.pseudo_npz)
    if "Y_hat" not in pseudo:
        raise KeyError(f"pseudo_npz must contain 'Y_hat'. keys={list(pseudo.keys())}")
    Y_hat = pseudo["Y_hat"].astype(np.int64)  # [N,T]
    conf = _load_conf(pseudo).astype(np.float32)

    if Y_hat.shape != conf.shape:
        raise ValueError(f"Y_hat shape {Y_hat.shape} != conf shape {conf.shape}")
    if Y_hat.shape[0] != N:
        raise ValueError(f"pseudo N mismatch: Y_hat N={Y_hat.shape[0]} vs Z N={N}")

    # stride pseudo to match Tp
    Y_hat_s = Y_hat[:, ::stride]
    conf_s = conf[:, ::stride]
    if Y_hat_s.shape[1] != Tp:
        # This can happen if Z and pseudo were dumped with different window length / concat_k
        raise ValueError(f"Tp mismatch: Z stride->Tp={Tp} but pseudo stride->Tp={Y_hat_s.shape[1]}")

    y_pseudo_flat = Y_hat_s.reshape(-1)       # [N*Tp]
    conf_flat = conf_s.reshape(-1)            # [N*Tp]

    y_pseudo_used = y_pseudo_flat[flat_idx]   # [M]
    conf_used = conf_flat[flat_idx]           # [M]

    # ----------------------------
    # Preprocess features (optional)
    # ----------------------------
    X_proc = X_feat
    if args.standardize:
        X_proc = StandardScaler().fit_transform(X_proc)
    if args.pca_dim and args.pca_dim > 0 and args.pca_dim < X_proc.shape[1]:
        X_proc = PCA(n_components=args.pca_dim, random_state=0).fit_transform(X_proc)

    # ----------------------------
    # Baseline on SAME points (no filtering)
    # ----------------------------
    print("\n===== Baseline on SAME points (feature space) =====")
    m_bag = _safe_metrics(X_proc, y_bag_used)
    m_pse = _safe_metrics(X_proc, y_pseudo_used)

    def _fmt(name, m):
        sil = "NA" if m["sil"] is None else f"{m['sil']:.4f}"
        db  = "NA" if m["db"]  is None else f"{m['db']:.4f}"
        ch  = "NA" if m["ch"]  is None else f"{m['ch']:.4f}"
        return f"[{name}] n={m['n']}, k={m['k']} | Sil↑ {sil} | DB↓ {db} | CH↑ {ch}"

    print(_fmt("Bag label", m_bag))
    print(_fmt("Pseudo label", m_pse))

    # ----------------------------
    # Confidence filtering sweep
    #   - For each tau, restrict BOTH labels AND features to SAME kept points
    # ----------------------------
    print("\n===== Confidence >= tau filtering (feature space) =====")
    print("tau | kept_n | kept_% | k | Sil↑ | DB↓ | CH↑ | ΔSil(inst-bag) | ΔDB(inst-bag) | ΔCH(inst-bag)")
    print("-" * 110)

    for tau in args.tau_list:
        keep = (conf_used >= float(tau))
        kept_n = int(keep.sum())
        kept_pct = 100.0 * kept_n / float(M)

        if kept_n < 10:
            print(f"{tau:>3.2f} | {kept_n:>6d} | {kept_pct:>6.2f}% |  - |  NA  |  NA  |   NA   |    NA        |     NA       |     NA")
            continue

        Xk = X_proc[keep]
        yb = y_bag_used[keep]
        yp = y_pseudo_used[keep]

        mb = _safe_metrics(Xk, yb)
        mp = _safe_metrics(Xk, yp)

        sil_b, sil_p = mb["sil"], mp["sil"]
        db_b,  db_p  = mb["db"],  mp["db"]
        ch_b,  ch_p  = mb["ch"],  mp["ch"]

        def _v(v):
            return "NA" if v is None else f"{v:.4f}"

        d_sil = None if (sil_b is None or sil_p is None) else (sil_p - sil_b)
        d_db  = None if (db_b  is None or db_p  is None) else (db_p  - db_b)   # lower is better
        d_ch  = None if (ch_b  is None or ch_p  is None) else (ch_p  - ch_b)

        print(
            f"{tau:>3.2f} | {kept_n:>6d} | {kept_pct:>6.2f}% | {mp['k']:>1d} | "
            f"{_v(sil_p):>6} | {_v(db_p):>6} | {_v(ch_p):>7} | "
            f"{('NA' if d_sil is None else f'{d_sil:+.4f}'):>12} | "
            f"{('NA' if d_db  is None else f'{d_db:+.4f}'):>11} | "
            f"{('NA' if d_ch  is None else f'{d_ch:+.4f}'):>11}"
        )


if __name__ == "__main__":
    main()
