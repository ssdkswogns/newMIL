#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate MILLET on the DBA dataset (bag metrics + instance accuracy for mixed).
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from dba_dataloader import (
    DBA_STYLE_TO_LABEL,
    build_dba_for_timemil,
    build_dba_windows_for_mixed,
)
from syntheticdataset import MixedSyntheticBagsConcatK
from models.milet import MILLET


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_testset(
    args: argparse.Namespace,
) -> Tuple[torch.utils.data.Dataset, int, int, int]:
    if args.datatype == "mixed":
        Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, feat_in = build_dba_windows_for_mixed(args)
        testset = MixedSyntheticBagsConcatK(
            X=Xte,
            y_idx=yte_idx,
            num_classes=num_classes,
            total_bags=len(Xte),
            concat_k=args.dba_concat_k,
            seed=args.seed + 1,
            return_instance_labels=True,
        )
        return testset, seq_len, num_classes, feat_in

    _, testset, seq_len, num_classes, feat_in = build_dba_for_timemil(args)
    return testset, seq_len, num_classes, feat_in


def evaluate(
    testloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, float], Optional[float]]:
    model.eval()
    total_loss = 0.0
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    inst_total_correct = 0
    inst_total_count = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(testloader):
            y_inst = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    feats, label, y_inst = batch
                elif len(batch) == 2:
                    feats, label = batch
                else:
                    raise ValueError(f"Unexpected batch length {len(batch)} in testloader.")
            else:
                raise ValueError(f"Unexpected batch type {type(batch)} in testloader.")

            bag_feats = feats.to(next(model.parameters()).device)
            bag_label = label.to(next(model.parameters()).device)

            out = model(bag_feats)
            if not isinstance(out, (tuple, list)) or len(out) < 3:
                raise ValueError("Unexpected MILLET output")
            bag_logits, instance_pred, _ = out

            loss = criterion(bag_logits, bag_label)
            total_loss += loss.item()

            probs = torch.sigmoid(bag_logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

            if y_inst is not None and instance_pred is not None:
                y_inst_label = torch.argmax(y_inst, dim=2).to(instance_pred.device)  # [B, T]
                pred_inst = torch.argmax(instance_pred, dim=2)  # [B, T]
                correct = (pred_inst == y_inst_label).sum().item()
                count = y_inst_label.numel()
                inst_total_correct += correct
                inst_total_count += count

            if batch_id % 10 == 0:
                print(f"\r Testing batch [{batch_id}/{len(testloader)}] loss: {loss.item():.4f}", end="")

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= args.cls_threshold).astype(np.int32)

    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    roc_list, ap_list = [], []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) == 2:
            try:
                roc_list.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
                ap_list.append(average_precision_score(y_true[:, c], y_prob[:, c]))
            except Exception:
                pass
    roc_macro = float(np.mean(roc_list)) if roc_list else 0.0
    ap_macro = float(np.mean(ap_list)) if ap_list else 0.0

    bag_acc = None
    if args.datatype == "original":
        true_cls = y_true.argmax(axis=1)
        pred_cls = y_prob.argmax(axis=1)
        bag_acc = float((true_cls == pred_cls).mean())

    inst_acc = None
    if inst_total_count > 0:
        inst_acc = float(inst_total_correct) / float(inst_total_count)

    results: Dict[str, float] = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "p_micro": p_micro,
        "p_macro": p_macro,
        "r_micro": r_micro,
        "r_macro": r_macro,
        "roc_auc_macro": roc_macro,
        "mAP_macro": ap_macro,
    }
    if bag_acc is not None:
        results["bag_acc"] = bag_acc

    return total_loss / max(1, len(testloader)), results, inst_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="MILLET evaluation on DBA")
    parser.add_argument("--model_path", required=True, type=str, help="Checkpoint path")
    parser.add_argument("--datatype", default="mixed", type=str, help="mixed | original")
    parser.add_argument("--dba_root", type=str, default="/home/giyong/newMIL/data/dba_parser")
    parser.add_argument("--dba_window", type=int, default=12000)
    parser.add_argument("--dba_stride", type=int, default=6000)
    parser.add_argument("--dba_test_ratio", type=float, default=0.2)
    parser.add_argument("--dba_concat_k", type=int, default=2)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--dropout_node", type=float, default=0.1)
    parser.add_argument("--millet_pooling", type=str, default="conjunctive")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cls_threshold", type=float, default=0.5)

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset, seq_len, num_classes, feat_in = build_testset(args)
    testloader = DataLoader(
        testset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = MILLET(
        feats_size=feat_in,
        mDim=args.embed,
        n_classes=num_classes,
        dropout=args.dropout_node,
        max_seq_len=seq_len,
        pooling=args.millet_pooling,
        is_instance=True,
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    criterion = nn.BCEWithLogitsLoss()
    test_loss, results, inst_acc = evaluate(testloader, model, criterion, args)

    class_names = list(DBA_STYLE_TO_LABEL.keys())
    print("\n===== DBA Evaluation (MILLET) =====")
    print(f"Checkpoint: {args.model_path}")
    print(f"Classes: {class_names}")
    print(f"Test loss: {test_loss:.6f}")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
    if inst_acc is not None:
        print(f"inst_acc: {inst_acc:.6f}")
    else:
        print("inst_acc: N/A (no instance labels)")


if __name__ == "__main__":
    main()
