#!/usr/bin/env python3
"""
Compare MILLET vs AmbiguousMIL embeddings with a shared autoencoder + t-SNE.
- Loads test split (aeon) -> MixedSyntheticBagsConcatK (return_instance_labels=True)
- Extracts timestep embeddings per model (backbone embeddings, not logits)
- Trains a small shared AE on the combined embeddings
- Projects latent to 2D with t-SNE; colors = class, markers = model
"""
import argparse
import random
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

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


def collect_backbone_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    model_name: str,
    sample_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (embeddings[N,D], labels[N], prototypes[C,D]) using backbone timestep embeddings.
    - AmbiguousMIL: inst_emb (out[5]) [B,T,D]
    - MILLET: feature_proj(feature_extractor(x)) [B,mDim,T] -> transpose to [B,T,D]
    """
    emb_list: List[torch.Tensor] = []
    lbl_list: List[torch.Tensor] = []
    proto_sum = None
    proto_cnt = None

    with torch.no_grad():
        for feats, _, y_inst in loader:
            feats = feats.to(device)
            y_inst = y_inst.to(device)

            if model_name == "AmbiguousMIL":
                out = model(feats)
                inst_emb = out[5].squeeze(0)  # [T,D]
            elif model_name == "MILLET":
                x = feats.transpose(1, 2)
                features = model.feature_extractor(x)
                features = model.feature_proj(features)
                inst_emb = features.transpose(1, 2).squeeze(0)  # [T,D]
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            y_inst_lbl = torch.argmax(y_inst.squeeze(0), dim=1)  # [T]

            # init proto buffers on first batch
            if proto_sum is None:
                proto_sum = torch.zeros(num_classes, inst_emb.shape[-1], device=device)
                proto_cnt = torch.zeros(num_classes, device=device)

            emb_list.append(inst_emb.cpu())
            lbl_list.append(y_inst_lbl.cpu())

            for c in range(num_classes):
                mask = y_inst_lbl == c
                if mask.any():
                    proto_sum[c] += inst_emb[mask].mean(dim=0)
                    proto_cnt[c] += 1

    emb_all = torch.cat(emb_list, dim=0)  # [N,D]
    lbl_all = torch.cat(lbl_list, dim=0)  # [N]

    # optional subsampling
    if sample_size > 0 and emb_all.shape[0] > sample_size:
        idx = torch.randperm(emb_all.shape[0])[:sample_size]
        emb_all = emb_all[idx]
        lbl_all = lbl_all[idx]

    proto = proto_sum / proto_cnt.clamp(min=1).unsqueeze(1)
    return emb_all.numpy(), lbl_all.numpy(), proto.cpu().numpy()


class SmallAE(nn.Module):
    def __init__(self, d_in: int, d_latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.ReLU(),
            nn.Linear(128, d_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_ae(embeddings: np.ndarray, d_latent: int, seed: int, device: torch.device, epochs: int):
    seed_all(seed)
    x = torch.from_numpy(embeddings).float()
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = SmallAE(x.shape[1], d_latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    model.train()
    for _ in range(epochs):  # lightweight
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            x_hat, _ = model(batch)
            loss = mse(x_hat, batch)
            loss.backward()
            opt.step()
    return model


def project_2d(latent: np.ndarray, proto_latent: np.ndarray, perplexity: float):
    try:
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0)
        all_pts = np.vstack([latent, proto_latent])
        pts_2d = reducer.fit_transform(all_pts)
    except Exception:
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto")
        all_pts = np.vstack([latent, proto_latent])
        pts_2d = tsne.fit_transform(all_pts)
    latent_2d = pts_2d[: len(latent)]
    proto_2d = pts_2d[len(latent) :]
    return latent_2d, proto_2d


def plot_tsne(latent_2d, labels, proto_2d, proto_labels, class_names, model_tags, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 7))
    markers = {0: "o", 1: "x"}  # 0=AmbiguousMIL, 1=MILLET
    for c in range(len(class_names)):
        for tag, mk in markers.items():
            mask = (labels == c) & (model_tags == tag)
            if mask.any():
                plt.scatter(
                    latent_2d[mask, 0],
                    latent_2d[mask, 1],
                    s=8,
                    alpha=0.6,
                    marker=mk,
                    label=f"{class_names[c]} (model {tag})",
                )

    # prototypes: star marker, colored by class text (offset to avoid overlap)
    for i, (x, y) in enumerate(proto_2d):
        plt.scatter(x, y, marker="*", s=180, edgecolors="k", linewidths=0.8, c="white", zorder=5)
        plt.text(x + 1.5, y + 1.5, str(class_names[proto_labels[i]]),
                 fontsize=8, ha="left", va="bottom", zorder=6,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2))

    # dedup legend
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

    plt.title("AE latent 2D (class color, marker: o=AmbiguousMIL, x=MILLET)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_per_model(latent_2d, labels, proto_2d, proto_labels, class_names, model_tags, out_prefix):
    import matplotlib.pyplot as plt

    for tag, mk in {0: "o", 1: "x"}.items():
        plt.figure(figsize=(9, 7))
        mask_model = model_tags == tag
        for c in range(len(class_names)):
            mask = mask_model & (labels == c)
            if mask.any():
                plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], s=10, alpha=0.7, marker=mk, label=class_names[c])
        # prototypes (해당 모델만 표기)
        proto_slice = proto_2d[tag * len(class_names) : (tag + 1) * len(class_names)]
        for i, (x, y) in enumerate(proto_slice):
            plt.scatter(x, y, marker="*", s=160, edgecolors="k", c="none")
            plt.text(x, y, str(class_names[i]), fontsize=7, ha="center", va="center")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        plt.title(f"AE latent 2D (모델 {tag})")
        plt.tight_layout()
        out_path = out_prefix.replace(".png", f"_model{tag}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="AE-based embedding compare for AmbiguousMIL vs MILLET")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--amb_model_path", type=str, required=True)
    parser.add_argument("--mil_model_path", type=str, required=True)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--dropout_node", type=float, default=0.2)
    parser.add_argument("--millet_pooling", type=str, default="conjunctive")
    parser.add_argument("--class_order", type=str, default="unique", choices=["unique", "meta"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=3000, help="timesteps per model to sample (0=all)")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--ae_epochs", type=int, default=25, help="AE training epochs")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--separate_plots", action="store_true", help="also save per-model plots")
    parser.add_argument("--output", type=str, default="ae_tsne_compare.png")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset, class_names, num_classes, feat_dim = load_aeon_dataset(
        name=args.dataset, extract_path="./data", seed=args.seed, class_order=args.class_order
    )
    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    # AmbiguousMIL model
    amb_model = AmbiguousMILwithCL(
        in_features=feat_dim,
        n_classes=num_classes,
        mDim=args.embed,
        dropout=args.dropout_node,
        is_instance=True,
    ).to(device)
    amb_state = torch.load(args.amb_model_path, map_location=device)
    amb_model.load_state_dict(amb_state)
    amb_model.eval()

    # MILLET model
    mil_model = MILLET(
        feat_dim,
        mDim=args.embed,
        n_classes=num_classes,
        dropout=args.dropout_node,
        max_seq_len=getattr(testset, "max_len", None) or testset[0][0].shape[0],
        pooling=args.millet_pooling,
        is_instance=True,
    ).to(device)
    mil_state = torch.load(args.mil_model_path, map_location=device)
    mil_model.load_state_dict(mil_state)
    mil_model.eval()

    amb_emb, amb_lbl, amb_proto = collect_backbone_embeddings(
        amb_model, loader, device, num_classes, "AmbiguousMIL", sample_size=args.sample_size
    )
    mil_emb, mil_lbl, mil_proto = collect_backbone_embeddings(
        mil_model, loader, device, num_classes, "MILLET", sample_size=args.sample_size
    )

    # stack for AE training (L2 normalize to align scales)
    def _l2norm(x: np.ndarray):
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / denom

    amb_emb = _l2norm(amb_emb)
    mil_emb = _l2norm(mil_emb)
    amb_proto = _l2norm(amb_proto)
    mil_proto = _l2norm(mil_proto)

    all_emb = np.vstack([amb_emb, mil_emb])
    model_tags = np.concatenate([np.full(len(amb_emb), 0), np.full(len(mil_emb), 1)])
    all_lbl = np.concatenate([amb_lbl, mil_lbl])

    ae = train_ae(all_emb, d_latent=args.latent_dim, seed=args.seed, device=device, epochs=args.ae_epochs)
    ae.eval()
    with torch.no_grad():
        z_all = ae.encoder(torch.from_numpy(all_emb).float().to(device)).cpu().numpy()
        z_proto_amb = ae.encoder(torch.from_numpy(amb_proto).float().to(device)).cpu().numpy()
        z_proto_mil = ae.encoder(torch.from_numpy(mil_proto).float().to(device)).cpu().numpy()

    # combine prototypes (keep label info)
    proto_latent = np.vstack([z_proto_amb, z_proto_mil])
    proto_labels = np.concatenate([np.arange(num_classes), np.arange(num_classes)])

    latent_2d, proto_2d = project_2d(z_all, proto_latent, perplexity=args.perplexity)
    plot_tsne(latent_2d, all_lbl, proto_2d, proto_labels, class_names, model_tags, args.output)


if __name__ == "__main__":
    main()
