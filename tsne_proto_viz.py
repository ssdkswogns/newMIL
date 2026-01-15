#!/usr/bin/env python3
"""
t-SNE visualization for AmbiguousMIL / MILLET instance embeddings with class prototypes.
Collects timestep embeddings, computes simple class prototypes, and plots both.
"""
import argparse
import random
import warnings
from typing import List, Tuple

import numpy as np
import torch
from sklearn.manifold import TSNE
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
    ytr_idx = torch.tensor([word_to_idx[i] for i in ytr], dtype=torch.long)
    yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)

    Xtr = torch.from_numpy(Xtr_np).permute(0, 2, 1).float()
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
    return testset, class_names, len(class_names), Xtr.shape[-1]


def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    model_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (inst_embeddings[N,D], inst_labels[N], proto[C,D]).
    Embeddings are chosen to align with instance acc computation:
      - AmbiguousMIL: weighted_instance_logits (B,T,C) -> embed dim=C
      - MILLET: interpretation (weighted instance logits) (B,T,C) -> embed dim=C
    """
    emb_list: List[torch.Tensor] = []
    lbl_list: List[torch.Tensor] = []
    proto_sum = torch.zeros(num_classes, num_classes, device=device)  # proto in logit space
    proto_cnt = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for feats, bag_lab, y_inst in loader:
            feats = feats.to(device)
            bag_lab = bag_lab.to(device)
            y_inst = y_inst.to(device)

            if model_name == "AmbiguousMIL":
                out = model(feats)
                # weighted instance logits: [B, T, C] (interpretation)
                inst_emb = out[2].squeeze(0)  # [T, C]
            elif model_name == "MILLET":
                # interpretation already weighted logits: [B, C, T]
                bag_logits, instance_pred, interpretation = model(feats)
                inst_emb = interpretation.squeeze(0).transpose(0, 1)  # [T,C]
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            # instance labels (one-hot)
            y_inst_lbl = torch.argmax(y_inst.squeeze(0), dim=1)  # [T]
            emb_list.append(inst_emb.cpu())
            lbl_list.append(y_inst_lbl.cpu())

            # proto accumulate using instance labels (if available), else bag label
            for c in range(num_classes):
                mask = y_inst_lbl == c
                if mask.any():
                    proto_sum[c] += inst_emb[mask].mean(dim=0)
                    proto_cnt[c] += 1

    inst_emb_all = torch.cat(emb_list, dim=0)  # [N,D]
    inst_lbl_all = torch.cat(lbl_list, dim=0)  # [N]
    proto = proto_sum / proto_cnt.clamp(min=1).unsqueeze(1)
    return inst_emb_all, inst_lbl_all, proto.cpu()


def run_tsne(emb: np.ndarray, proto: np.ndarray, labels: np.ndarray, sample_size: int, perplexity: float):
    n = emb.shape[0]
    idx = np.random.permutation(n)[: min(sample_size, n)]
    emb_sample = emb[idx]
    lbl_sample = labels[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(np.vstack([emb_sample, proto]))
    emb_pts = emb_2d[: len(emb_sample)]
    proto_pts = emb_2d[len(emb_sample) :]
    return emb_pts, lbl_sample, proto_pts


def plot_tsne(emb_pts, lbl_sample, proto_pts, class_names, out_path: str):
    import matplotlib.pyplot as plt

    num_classes = len(class_names)
    plt.figure(figsize=(7, 6))
    for c in range(num_classes):
        mask = lbl_sample == c
        if mask.any():
            plt.scatter(emb_pts[mask, 0], emb_pts[mask, 1], s=8, alpha=0.5, label=str(class_names[c]))
    plt.scatter(proto_pts[:, 0], proto_pts[:, 1], c="k", marker="*", s=120, edgecolors="w", label="prototype")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.title("Instance t-SNE with prototypes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved t-SNE plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization with prototypes (AmbiguousMIL/MILLET)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["AmbiguousMIL", "MILLET"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--dropout_node", type=float, default=0.2)
    parser.add_argument("--millet_pooling", type=str, default="conjunctive")
    parser.add_argument("--class_order", type=str, default="unique", choices=["unique", "meta"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=2000, help="number of instances to sample for t-SNE")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output", type=str, default="tsne_proto.png")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset/model
    testset, class_names, num_classes, feat_dim = load_aeon_dataset(
        name=args.dataset, extract_path="./data", seed=args.seed, class_order=args.class_order
    )

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

    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    inst_emb_all, inst_lbl_all, proto = collect_embeddings(model, loader, device, num_classes, args.model)
    emb_pts, lbl_sample, proto_pts = run_tsne(
        inst_emb_all.numpy(), proto.numpy(), inst_lbl_all.numpy(), sample_size=args.sample_size, perplexity=args.perplexity
    )
    plot_tsne(emb_pts, lbl_sample, proto_pts, class_names, args.output)


if __name__ == "__main__":
    main()
