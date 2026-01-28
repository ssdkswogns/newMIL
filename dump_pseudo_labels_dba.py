# -*- coding: utf-8 -*-
"""
Dump AmbiguousMIL per-timestep hard pseudo labels for DBA.

Output:
  - Y_hat: [N, T] int64   (per-timestep argmax class id)
Optional:
  - conf:  [N, T] float32 (per-timestep max probability/logit-derived score)

IMPORTANT:
- dataset build args(datatype/window/stride/test_ratio/concat_k/seed 등)는
  AE embeddings를 덤프할 때와 반드시 동일해야 합니다.
- inference시 shuffle=False 고정.

Usage:
python dump_pseudo_labels_dba.py \
  --ckpt ./savemodel/InceptBackbone/dba/exp_x/weights/best_AmbiguousMIL.pth \
  --datatype mixed \
  --dba_root ./data/dba_data --dba_window 12000 --dba_stride 6000 --dba_test_ratio 0.2 \
  --concat_k 2 \
  --embed 128 \
  --batchsize 8 \
  --out_npz ./tsne_out/pseudo_inst_labels.npz
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from dba_dataloader import build_dba_for_timemil, build_dba_windows_for_mixed
from syntheticdataset import MixedSyntheticBagsConcatK
from models.expmil import AmbiguousMILwithCL
from models.millet import MILLET
from models.timemil import TimeMIL, originalTimeMIL


def maybe_mkdir_p(p):
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Dump per-timestep pseudo labels (AmbiguousMIL) for DBA")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--datatype", type=str, default="mixed", choices=["original", "mixed"])
    parser.add_argument('--model', default='TimeMIL', type=str, help='MIL model')

    # DBA args (AE와 동일하게)
    parser.add_argument("--dba_root", type=str, default="./data/dba_data")
    parser.add_argument("--dba_window", type=int, default=12000)
    parser.add_argument("--dba_stride", type=int, default=6000)
    parser.add_argument("--dba_test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concat_k", type=int, default=2)

    # model args
    parser.add_argument("--embed", type=int, default=128, help="mDim used in AmbiguousMIL")
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    # output
    parser.add_argument("--out_npz", type=str, required=True)
    parser.add_argument("--save_conf", action="store_true", help="also save per-timestep confidence")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- build dataset (same as AE dump) ----
    if args.datatype == "original":
        trainset, testset, seq_len, num_classes, in_dim = build_dba_for_timemil(args)
    else:
        Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, in_dim = build_dba_windows_for_mixed(args)
        trainset = MixedSyntheticBagsConcatK(
            X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
            total_bags=len(Xtr), concat_k=args.concat_k, seed=args.seed,
            return_instance_labels=False
        )
        testset = MixedSyntheticBagsConcatK(
            X=Xte, y_idx=yte_idx, num_classes=num_classes,
            total_bags=len(Xte), concat_k=args.concat_k, seed=args.seed + 1,
            return_instance_labels=True
        )

    full = ConcatDataset([trainset, testset])

    loader = DataLoader(
        full,
        batch_size=args.batchsize,
        shuffle=False,             # ★ 중요
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    if args.model == 'AmbiguousMIL':
        model = AmbiguousMILwithCL(
            in_features=in_dim,
            n_classes=num_classes,
            mDim=args.embed,
            dropout=0.0,
            is_instance=True,
        ).to(device)
    elif args.model == 'MILLET':
        model = MILLET(feats_size=in_dim, mDim=args.embed, n_classes=num_classes,
            dropout=0.0, max_seq_len=seq_len,
            pooling='conjunctive', is_instance=True).to(device)
    elif args.model == 'newTimeMIL':
        model = TimeMIL(feats_size=in_dim, mDim=args.embed, n_classes=num_classes, 
            dropout=0.0, max_seq_len=seq_len, is_instance=True).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    Y_list = []
    C_list = []

    for batch in loader:
        # batch could be (feats, label) or (feats, label, y_inst)
        feats = batch[0].to(device).float()  # [B,T,D]
        if args.model == 'AmbiguousMIL':
            out = model(feats)
            # AmbiguousMILwithCL returns:
            # (bag_logits_ts, bag_logits_tp, weighted_logits, inst_logits, bag_emb, inst_emb, attn_ts, attn_tp)
            weighted_logits = out[2]  # [B,T,C]
            inst_logits = out[3]      # [B,T,C]
        elif args.model == 'MILLET':
            out = model(feats)
            bag_prediction, inst_logits, interpretation = out
            weighted_logits = interpretation.transpose(1, 2)  # [B,T,C]
        elif args.model == 'newTimeMIL':
            out = model(feats)
            logits, x_cls, attn_layer1, attn_layer2 = out
            attn_cls2time = attn_layer2[:, :num_classes, num_classes:]
            weighted_logits = attn_cls2time.transpose(1, 2)  # [B,T,C]

        # hard label: argmax over class dimension
        y_hat = torch.argmax(weighted_logits, dim=-1)  # [B,T]
        Y_list.append(y_hat.cpu().numpy().astype(np.int64))

        if args.save_conf:
            # confidence: max softmax probability from weighted_logits
            p = F.softmax(weighted_logits, dim=-1)          # [B,T,C]
            conf = torch.max(p, dim=-1).values              # [B,T]
            C_list.append(conf.cpu().numpy().astype(np.float32))

    Y_hat = np.concatenate(Y_list, axis=0)  # [N,T]
    maybe_mkdir_p(os.path.dirname(args.out_npz))

    if args.save_conf:
        conf = np.concatenate(C_list, axis=0)
        np.savez_compressed(args.out_npz, Y_hat=Y_hat, conf=conf)
        print(f"[save] {args.out_npz} | Y_hat={Y_hat.shape}, conf={conf.shape}")
    else:
        np.savez_compressed(args.out_npz, Y_hat=Y_hat)
        print(f"[save] {args.out_npz} | Y_hat={Y_hat.shape}")

    print("[done]")


if __name__ == "__main__":
    main()
