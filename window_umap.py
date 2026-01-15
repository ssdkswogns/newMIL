#!/usr/bin/env python3
"""
Sliding-window embedding visualization (UMAP fallback to t-SNE) for AmbiguousMIL / MILLET.
- Extract backbone timestep embeddings
- Aggregate with sliding windows to keep temporal locality
- Project to 2D (UMAP if available, else t-SNE)
- Color by class, mark class prototypes (window embeddings averaged per class)
"""
import argparse
import random
import warnings
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from aeon.datasets import load_classification
from syntheticdataset import MixedSyntheticBagsConcatK
from models.AmbiguousMIL import AmbiguousMILwithCL
from models.milet import MILLET

warnings.filterwarnings("ignore")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_aeon_dataset(name: str, extract_path: str, seed: int, class_order: str):
    res_tr = load_classification(name=name, split="train", extract_path=extract_path, return_metadata=True)
    res_te = load_classification(name=name, split="test", extract_path=extract_path, return_metadata=True)

    if len(res_tr) == 3:
        Xtr_np, ytr, meta = res_tr
    else:
        Xtr_np, ytr = res_tr
        meta = {"class_values": np.unique(ytr)}

    if len(res_te) == 3:
        Xte_np, yte, _ = res_te
    else:
        Xte_np, yte = res_te

    if class_order == "meta" and meta.get("class_values") is not None:
        class_values = meta["class_values"]
    else:
        class_values = np.unique(ytr)

    class_names = list(class_values)
    word_to_idx = {cls: i for i, cls in enumerate(class_values)}
    yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)

    Xte = torch.from_numpy(Xte_np).permute(0, 2, 1).float()
    seed_all(seed)
    testset = MixedSyntheticBagsConcatK(
        X=Xte,
        y_idx=yte_idx,
        num_classes=len(class_values),
        total_bags=len(Xte),
        seed=seed + 1,
        return_instance_labels=True,
    )
    return testset, class_names, len(class_names), Xte.shape[-1]


def get_backbone_embeddings(model, feats, model_name):
    if model_name == "AmbiguousMIL":
        out = model(feats)
        inst_emb = out[5]  # [B,T,D]
    elif model_name == "MILLET":
        x = feats.transpose(1, 2)
        features = model.feature_extractor(x)
        features = model.feature_proj(features)
        inst_emb = features.transpose(1, 2)  # [B,T,D]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return inst_emb


def sliding_windows(emb: torch.Tensor, labels: torch.Tensor, win: int, stride: int):
    """
    emb: [T,D], labels: [T]
    returns list of (window_emb[D], window_label[int])
    """
    T, D = emb.shape
    out_emb, out_lbl = [], []
    for start in range(0, T - win + 1, stride):
        chunk = emb[start : start + win]  # [win,D]
        chunk_lbl = labels[start : start + win]
        window_emb = chunk.mean(dim=0)
        # majority label
        vals, counts = chunk_lbl.unique(return_counts=True)
        maj = vals[counts.argmax()].item()
        out_emb.append(window_emb)
        out_lbl.append(maj)
    return out_emb, out_lbl


def collect_windows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    model_name: str,
    win: int,
    stride: int,
    sample_size: int,
):
    emb_list: List[torch.Tensor] = []
    lbl_list: List[int] = []
    proto_sum = torch.zeros(num_classes, model.mDim if hasattr(model, "mDim") else model.n_classes, device=device)
    proto_cnt = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for feats, _, y_inst in loader:
            feats = feats.to(device)
            y_inst = y_inst.to(device)

            inst_emb = get_backbone_embeddings(model, feats, model_name).squeeze(0)  # [T,D]
            y_lbl = torch.argmax(y_inst.squeeze(0), dim=1)  # [T]

            ws_emb, ws_lbl = sliding_windows(inst_emb, y_lbl, win, stride)
            if not ws_emb:
                continue
            ws_emb_t = torch.stack(ws_emb)  # [Nw,D]
            ws_lbl_t = torch.tensor(ws_lbl, device=device)

            emb_list.append(ws_emb_t.cpu())
            lbl_list.append(ws_lbl_t.cpu())

            for c in range(num_classes):
                mask = ws_lbl_t == c
                if mask.any():
                    proto_sum[c] += ws_emb_t[mask].mean(dim=0).to(device)
                    proto_cnt[c] += 1

    if not emb_list:
        raise RuntimeError("No windows collected; check window size/stride.")

    emb_all = torch.cat(emb_list, dim=0)  # [N,D]
    lbl_all = torch.cat(lbl_list, dim=0)  # [N]

    if sample_size > 0 and emb_all.shape[0] > sample_size:
        idx = torch.randperm(emb_all.shape[0])[:sample_size]
        emb_all = emb_all[idx]
        lbl_all = lbl_all[idx]

    proto = proto_sum / proto_cnt.clamp(min=1).unsqueeze(1)
    return emb_all.numpy(), lbl_all.numpy(), proto.cpu().numpy()


def project_2d(emb: np.ndarray, proto: np.ndarray, perplexity: float):
    try:
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0)
        all_pts = np.vstack([emb, proto])
        pts_2d = reducer.fit_transform(all_pts)
    except Exception:
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto")
        all_pts = np.vstack([emb, proto])
        pts_2d = tsne.fit_transform(all_pts)
    emb_2d = pts_2d[: len(emb)]
    proto_2d = pts_2d[len(emb) :]
    return emb_2d, proto_2d


def plot(emb_2d, labels, proto_2d, class_names, model_tag, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    num_classes = len(class_names)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=6, alpha=0.5, label=str(class_names[c]))
    plt.scatter(proto_2d[:, 0], proto_2d[:, 1], marker="*", s=140, edgecolors="k", c="none", label="prototype")
    plt.title(f"{model_tag} window embeddings (2D)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sliding-window UMAP/t-SNE for AmbiguousMIL / MILLET")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["AmbiguousMIL", "MILLET"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--dropout_node", type=float, default=0.2)
    parser.add_argument("--millet_pooling", type=str, default="conjunctive")
    parser.add_argument("--class_order", type=str, default="unique", choices=["unique", "meta"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=5000, help="max windows to visualize (0=all)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity fallback")
    parser.add_argument("--output", type=str, default="window_vis.png")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset, class_names, num_classes, feat_dim = load_aeon_dataset(
        name=args.dataset, extract_path="./data", seed=args.seed, class_order=args.class_order
    )
    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    if args.model == "AmbiguousMIL":
        model = AmbiguousMILwithCL(
            in_features=feat_dim,
            n_classes=num_classes,
            mDim=args.embed,
            dropout=args.dropout_node,
            is_instance=True,
        ).to(device)
    else:
        model = MILLET(
            feat_dim,
            mDim=args.embed,
            n_classes=num_classes,
            dropout=args.dropout_node,
            max_seq_len=getattr(testset, "max_len", None) or testset[0][0].shape[0],
            pooling=args.millet_pooling,
            is_instance=True,
        ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    emb, lbl, proto = collect_windows(
        model, loader, device, num_classes, args.model, args.window, args.stride, args.sample_size
    )
    emb_2d, proto_2d = project_2d(emb, proto, perplexity=args.perplexity)
    plot(emb_2d, lbl, proto_2d, class_names, args.model, args.output)


if __name__ == "__main__":
    main()
