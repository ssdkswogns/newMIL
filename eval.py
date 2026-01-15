# -*- coding: utf-8 -*-
"""
Combined evaluation script: computes classification metrics (bag + instance)
and class-wise AOPCR in a single run.
"""

import argparse
import os
import random
import sys
import warnings
from typing import Dict, List, Optional

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

from aeon.datasets import load_classification
from syntheticdataset import MixedSyntheticBagsConcatK
from utils import *
from mydataload import loadorean
from models.AmbiguousMIL import AmbiguousMILwithCL
from compute_aopcr import compute_classwise_aopcr
from models.milet import MILLET

warnings.filterwarnings("ignore")

# ------------------------ Seed ------------------------
# NOTE: Removed global seed initialization.
# Seed is set later from args.seed to allow different random test sets per evaluation.


def evaluate_classification(testloader, milnet, criterion, args, class_names, threshold: float = 0.5):
    milnet.eval()
    total_loss = 0.0
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    per_class_correct = None
    per_class_total = None
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch_id, (feats, label, y_inst) in enumerate(testloader):
            bag_feats = feats.to(next(milnet.parameters()).device)
            bag_label = label.to(next(milnet.parameters()).device)

            out = milnet(bag_feats)
            if args.model == 'AmbiguousMIL':
                if not isinstance(out, (tuple, list)) or len(out) < 4:
                    raise ValueError("Unexpected AmbiguousMIL output")
                bag_logits_ts, bag_logits_tp, weighted_logits, inst_logits = out[:4]
                # tp head is main bag prediction; weighted logits are per-timestep
                logits = bag_logits_tp
                instance_pred = weighted_logits
                attn_layer2 = None
            elif args.model == 'MILLET':
                if not isinstance(out, (tuple, list)) or len(out) < 3:
                    raise ValueError("Unexpected MILLET output")
                # MILLET returns (bag_logits, instance_logits[B,T,C], interpretation[B,C,T])
                logits, non_weighted_instance_pred, instance_pred = out
                attn_layer2 = None
            else:
                raise ValueError(f"Unknown model {args.model}")

            loss = criterion(logits, bag_label)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()  # [B, C]
            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

            _, _, C = y_inst.shape
            if args.model == 'MILLET' and instance_pred is not None:
                # MILLET interpretation map is [B, C, T]; follow main_cl_fix behavior
                pred_inst = torch.argmax(non_weighted_instance_pred, dim=2).cpu()
            else:
                if args.model == 'AmbiguousMIL' and instance_pred is not None:
                    pred_inst = torch.argmax(instance_pred, dim=2).cpu()
                else:
                    raise ValueError("Instance prediction not available for model")

            y_inst_label = torch.argmax(y_inst, dim=2).cpu()

            correct = (pred_inst == y_inst_label).sum().item()
            count = pred_inst.numel()

            total_correct += correct
            total_count += count

            if per_class_correct is None:
                per_class_correct = torch.zeros(C, dtype=torch.long)
                per_class_total = torch.zeros(C, dtype=torch.long)

            pb = pred_inst.view(-1)
            tb = y_inst_label.view(-1)
            for c_id in range(C):
                mask = (tb == c_id)
                if mask.any():
                    per_class_correct[c_id] += (pb[mask] == tb[mask]).sum().cpu()
                    per_class_total[c_id] += mask.sum().cpu()

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f'
                             % (batch_id, len(testloader), loss.item()))

    y_true = np.vstack(all_labels)   # [N, C], multi-hot
    y_prob = np.vstack(all_probs)    # [N, C], sigmoid prob
    y_pred = (y_prob >= threshold).astype(np.int32)  # [N, C]

    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    p_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

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

    timestep_acc = float(total_correct / total_count) if total_count > 0 else 0.0
    per_class_acc = []
    for c_id in range(per_class_total.numel()):
        tot = int(per_class_total[c_id].item())
        cor = int(per_class_correct[c_id].item())
        acc = float(cor / tot) if tot > 0 else 0.0
        label = class_names[c_id] if (class_names is not None and c_id < len(class_names)) else f"class-{c_id}"
        per_class_acc.append({"class_id": c_id, "label": label, "count": tot, "correct": cor, "acc": acc})

    inst_acc = {
        "timestep_accuracy_overall": timestep_acc,
        "per_class": per_class_acc
    }

    results: Dict[str, float] = {
        "f1_micro": f1_micro, "f1_macro": f1_macro,
        "p_micro": p_micro, "p_macro": p_macro,
        "r_micro": r_micro, "r_macro": r_macro,
        "roc_auc_macro": roc_macro, "mAP_macro": ap_macro
    }
    return total_loss / len(testloader), results, inst_acc


def build_datasets(args):
    """
    Mirrors the logic from compute_aopcr and instance_val_cl to build train/test sets.
    Returns (testset, class_names, seq_len, num_classes, feat_dim)
    """
    def _load_split_with_meta(name: str, split: str, extract_path: str):
        """aeon load_classification wrapper that always returns (X, y, meta)."""
        res = load_classification(
            name=name,
            split=split,
            extract_path=extract_path,
            return_metadata=True,
        )
        if isinstance(res, tuple):
            if len(res) == 3:
                return res
            if len(res) == 2:
                X, y = res
                meta = {"class_values": list(dict.fromkeys(y))}
                return X, y, meta
        raise ValueError(
            f"Unexpected return from load_classification for {name} ({split}): "
            f"{type(res)} len={len(res) if hasattr(res,'__len__') else 'na'}"
        )

    if args.dataset in ['JapaneseVowels', 'SpokenArabicDigits', 'CharacterTrajectories', 'InsectWingbeat']:
        trainset = loadorean(args, split='train')
        testset = loadorean(args, split='test')

        seq_len, num_classes, L_in = trainset.max_len, trainset.num_class, trainset.feat_in
        args.feats_size = L_in
        args.num_classes = num_classes

        if hasattr(trainset, 'class_names'):
            class_names = list(trainset.class_names)
        else:
            class_names = [str(i) for i in range(num_classes)]
        return testset, class_names, seq_len, num_classes, L_in

    Xtr, ytr, meta = _load_split_with_meta(name=args.dataset, split='train', extract_path='./data')
    Xte, yte, _ = _load_split_with_meta(name=args.dataset, split='test', extract_path='./data')

    Xtr = torch.from_numpy(Xtr).permute(0, 2, 1).float()  # [N,T,D]
    Xte = torch.from_numpy(Xte).permute(0, 2, 1).float()

    class_order = getattr(args, "class_order", "unique")
    if class_order == "meta" and meta.get("class_values", None) is not None:
        class_values = meta["class_values"]
    else:
        # default: np.unique to mirror training (lexicographic ordering for string labels)
        class_values = np.unique(ytr)
    num_classes = len(class_values)
    args.num_classes = num_classes

    class_names = list(class_values)
    word_to_idx = {cls: i for i, cls in enumerate(class_values)}
    yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)

    L_in = Xtr.shape[-1]
    seq_len = max(21, Xte.shape[1])
    args.feats_size = L_in

    # CRITICAL: Re-seed before dataset creation to ensure reproducibility
    # (aeon's load_classification may consume random state)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    testset = MixedSyntheticBagsConcatK(
        X=Xte, y_idx=yte_idx, num_classes=num_classes,
        total_bags=len(Xte),
        seed=args.seed + 1,
        return_instance_labels=True
    )

    print(f'num class: {args.num_classes}')
    print(f'total_len: {seq_len}')
    return testset, class_names, seq_len, num_classes, L_in


def build_model(args, seq_len, num_classes, device):
    if args.model == 'AmbiguousMIL':
        milnet = AmbiguousMILwithCL(
            in_features=args.feats_size,
            n_classes=num_classes,
            mDim=args.embed,
            dropout=args.dropout_node if hasattr(args, 'dropout_node') else 0.0,
            is_instance=True,
        ).to(device)
    elif args.model == 'MILLET':
        milnet = MILLET(args.feats_size, mDim=args.embed,
                        n_classes=num_classes,
                        dropout=args.dropout_node if hasattr(args, 'dropout_node') else 0.0,
                        max_seq_len=seq_len,
                        pooling=getattr(args, "millet_pooling", "conjunctive"),
                        is_instance=True).to(device)
    else:
        raise Exception("Model not available")
    return milnet


def main():
    parser = argparse.ArgumentParser(description='Compute classification metrics and AOPCR together')
    parser.add_argument('--dataset', default="BasicMotions", type=str, help='dataset')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--model', default='AmbiguousMIL', type=str,
                        help='MIL model: AmbiguousMIL | MILLET')
    parser.add_argument('--model_path', default='./savemodel/InceptBackbone/BasicMotions/exp_7/weights/best_AmbiguousMIL.pth',
                        type=str, help='Target model path for evaluation')
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--num_random', default=3, type=int, help='Random baseline repetition for AOPCR')
    parser.add_argument('--aopcr_stop', default=0.5, type=float,
                        help='AOPCR perturbation limit (0.0-1.0). Default 0.5 (50%%). Use 1.0 for full range.')
    parser.add_argument('--aopcr_step', default=0.05, type=float,
                        help='AOPCR perturbation step size. Default 0.05 (5%%).')
    parser.add_argument('--aopcr_save_npz', default=None, type=str,
                        help='Optional path to save AOPCR curves (npz with M_expl, M_rand, alphas, counts)')
    parser.add_argument('--plot_aopcr', action='store_true',
                        help='If set, save a weighted mean AOPCR curve plot (matplotlib).')
    parser.add_argument('--aopcr_plot_path', default='aopcr_curve.png', type=str,
                        help='Output path for AOPCR curve plot when --plot_aopcr is used.')
    parser.add_argument('--dropout_node', default=0.0, type=float, help='Dropout rate for classifier heads')
    parser.add_argument('--millet_pooling', default="conjunctive", type=str,
                        help="MILLET pooling method: conjunctive | attention | instance | additive | gap")
    parser.add_argument('--no_fallback_to_predicted', action='store_false', dest='fallback_to_predicted',
                        help='If set, skip bags with no positive labels instead of using argmax prediction')
    parser.add_argument('--cls_threshold', default=0.5, type=float, help='Threshold for multi-label classification metrics')
    parser.add_argument('--datatype', default="mixed", type=str, help='Choose datatype between original and mixed')
    parser.add_argument('--class_order', default="unique", choices=["unique", "meta"],
                        help='Class index order: "unique" (np.unique, matches training) or "meta" (meta.class_values order)')

    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed 재설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 데이터/모델 준비
    testset, class_names, seq_len, num_classes, feat_dim = build_datasets(args)
    milnet = build_model(args, seq_len, num_classes, device)

    # DataLoader (classification)
    testloader = DataLoader(
        testset, batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )

    # 모델 로드
    state = torch.load(args.model_path, map_location=device)
    milnet.load_state_dict(state)
    milnet.to(device)

    # 분류/인스턴스 성능
    criterion = nn.BCEWithLogitsLoss()
    print("Computing classification/instance metrics ...")
    test_loss_bag, cls_results, inst_acc = evaluate_classification(
        testloader, milnet, criterion, args, class_names, threshold=args.cls_threshold
    )
    print("\nClassification metrics:", cls_results)
    print("Instance accuracy:", inst_acc)

    # AOPCR (use batch_size=1 for perturbation stability)
    testloader_aopcr = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )

    # AOPCR
    print(f"Computing class-wise AOPCR (stop={args.aopcr_stop}, step={args.aopcr_step}) ...")
    result = compute_classwise_aopcr(
        milnet,
        testloader_aopcr,
        args,
        stop=args.aopcr_stop,
        step=args.aopcr_step,
        n_random=args.num_random,
        pred_threshold=0.5,
        coverage_thresholds=[0.9, 0.8, 0.7, 0.5]
    )
    aopcr_c, aopcr_w_avg, aopcr_mean, aopcr_overall, M_expl, M_rand, alphas, counts, coverage_summary = result

    if args.aopcr_save_npz:
        np.savez(
            args.aopcr_save_npz,
            M_expl=M_expl,
            M_rand=M_rand,
            alphas=alphas,
            counts=counts,
            aopcr_c=aopcr_c,
            aopcr_w_avg=aopcr_w_avg,
            aopcr_mean=aopcr_mean,
            coverage_summary=coverage_summary,
            allow_pickle=True,
        )
        print(f"Saved AOPCR arrays to {args.aopcr_save_npz}")

    if args.plot_aopcr:
        try:
            import matplotlib.pyplot as plt

            total = counts.sum()
            if total > 0:
                weights = counts / total
                curve_expl = (M_expl * weights[:, None]).sum(axis=0)
                curve_rand = (M_rand * weights[:, None]).sum(axis=0)
            else:
                curve_expl = M_expl.mean(axis=0)
                curve_rand = M_rand.mean(axis=0)

            plt.figure(figsize=(6, 4))
            plt.plot(alphas, curve_expl, label="explanation")
            plt.plot(alphas, curve_rand, label="random", linestyle="--")
            plt.xlabel("Perturbation ratio")
            plt.ylabel("Logit")
            plt.title("AOPCR curves (weighted mean logits)")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.aopcr_plot_path, dpi=200)
            plt.close()
            print(f"Saved AOPCR plot to {args.aopcr_plot_path}")
        except ImportError:
            print("matplotlib not installed; skipping AOPCR plot")

    print("\n===== Class-wise AOPCR =====")
    for c in range(args.num_classes):
        name = class_names[c] if c < len(class_names) else f"class-{c}"
        cnt = int(counts[c])
        val = aopcr_c[c]
        if np.isnan(val):
            print(f"[{c}] {name:>20s} | count={cnt:4d} | AOPCR = NaN (no positive bags)")
        else:
            print(f"[{c}] {name:>20s} | count={cnt:4d} | AOPCR = {val:.6f}")

    print("\n===== Average AOPCR =====")
    print(f"Weighted Average AOPCR: {aopcr_w_avg:.6f}")
    print(f"Mean AOPCR: {aopcr_mean:.6f}")
    if aopcr_overall is not None:
        print(f"Overall AOPCR (original dataset): {aopcr_overall:.6f}")

    # Coverage metrics
    print("\n===== Coverage Metrics =====")
    print("(Coverage@X = % of instances needed to maintain X% of original performance)")
    for thr in sorted(coverage_summary.keys(), reverse=True):
        cov = coverage_summary[thr]
        print(f"\nCoverage@{thr:.0%}:")
        print(f"  Explanation: {cov['expl_mean']:.4f} (weighted: {cov['expl_weighted']:.4f})")
        print(f"  Random:      {cov['rand_mean']:.4f} (weighted: {cov['rand_weighted']:.4f})")
        print(f"  Gain:        {cov['coverage_gain']:.4f} (explanation better than random)")


if __name__ == '__main__':
    main()
