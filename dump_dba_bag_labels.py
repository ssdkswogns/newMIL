# -*- coding: utf-8 -*-
"""
DBA dataset을 build해서 bag label만 [N]으로 저장.
AE 임베딩 Z의 N과 동일한 순서/개수여야 하므로,
AE 임베딩을 뽑을 때 사용한 것과 동일한 args(datatype/window/stride/test_ratio/seed 등)로 build하십시오.

Usage:
python dump_dba_bag_labels.py \
  --datatype mixed \
  --dba_root ./data/dba_data --dba_window 12000 --dba_stride 6000 --dba_test_ratio 0.2 \
  --concat_k 2 \
  --out_path ./savemodel_ae/dba_ae/y_bag.npy
"""

import os
import argparse
import numpy as np
import torch

from torch.utils.data import ConcatDataset

from dba_dataloader import build_dba_for_timemil, build_dba_windows_for_mixed
from syntheticdataset import MixedSyntheticBagsConcatK


def maybe_mkdir_p(p: str):
    os.makedirs(p, exist_ok=True)


def get_label_from_item(item):
    # item: (feats, label) or (feats, label, y_inst)
    if isinstance(item, (tuple, list)):
        y = item[1]
    else:
        raise ValueError("Dataset item must be tuple/list")
    if torch.is_tensor(y):
        y = y.detach().cpu()
        if y.ndim == 0:
            return int(y.item())
        if y.numel() > 1:
            return int(torch.argmax(y).item())
        return int(y.item())
    y = np.asarray(y)
    if y.ndim == 0:
        return int(y)
    return int(np.argmax(y))


def main():
    parser = argparse.ArgumentParser("Dump DBA bag labels")
    parser.add_argument("--datatype", type=str, default="mixed", choices=["original", "mixed"])
    parser.add_argument("--dba_root", type=str, default="./data/dba_data")
    parser.add_argument("--dba_window", type=int, default=12000)
    parser.add_argument("--dba_stride", type=int, default=6000)
    parser.add_argument("--dba_test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concat_k", type=int, default=2)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    if args.datatype == "original":
        trainset, testset, seq_len, num_classes, in_dim = build_dba_for_timemil(args)
    else:
        Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, in_dim = build_dba_windows_for_mixed(args)
        trainset = MixedSyntheticBagsConcatK(
            X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
            total_bags=len(Xtr), concat_k=args.concat_k, seed=args.seed,
            return_instance_labels=False,
        )
        testset = MixedSyntheticBagsConcatK(
            X=Xte, y_idx=yte_idx, num_classes=num_classes,
            total_bags=len(Xte), concat_k=args.concat_k, seed=args.seed + 1,
            return_instance_labels=True,
        )

    full = ConcatDataset([trainset, testset])
    y_list = [get_label_from_item(full[i]) for i in range(len(full))]
    y = np.asarray(y_list, dtype=np.int64)

    maybe_mkdir_p(os.path.dirname(args.out_path))
    np.save(args.out_path, y)
    print(f"[save] {args.out_path} | y={y.shape} | classes={np.unique(y)}")


if __name__ == "__main__":
    main()
