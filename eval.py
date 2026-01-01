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
from models.timemil import TimeMIL, newTimeMIL
from models.expmil import AmbiguousMILwithCL
from compute_aopcr import compute_classwise_aopcr

warnings.filterwarnings("ignore")

# ------------------------ Seed ------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
                if not isinstance(out, (tuple, list)) or len(out) < 6:
                    raise ValueError("Unexpected AmbiguousMIL output")
                logits, instance_pred, pred_inst = out[0], out[1], out[2]
                attn_layer2 = None
            else:
                if not isinstance(out, (tuple, list)) or len(out) < 4:
                    raise ValueError("Unexpected model output")
                logits, _, _, attn_layer2 = out
                attn_cls = attn_layer2[:, :, :C, C:]
                attn_mean = attn_cls.mean(dim=1)
                instance_pred = None

            loss = criterion(logits, bag_label)
            total_loss += loss.item()

            if args.model == 'AmbiguousMIL':
                probs = torch.sigmoid(instance_pred).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

            _, _, C = y_inst.shape

            if args.model == 'AmbiguousMIL' and instance_pred is not None:
                pred_inst = torch.argmax(pred_inst, dim=2).cpu()
            else:
                pred_inst = torch.argmax(attn_mean, dim=1).cpu()

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

    Xtr, ytr, meta = load_classification(name=args.dataset, split='train', extract_path='./data')
    Xte, yte, _ = load_classification(name=args.dataset, split='test', extract_path='./data')

    Xtr = torch.from_numpy(Xtr).permute(0, 2, 1).float()  # [N,T,D]
    Xte = torch.from_numpy(Xte).permute(0, 2, 1).float()

    class_values = meta.get('class_values', None)
    num_classes = len(class_values) if class_values is not None else len(np.unique(ytr))
    args.num_classes = num_classes

    if class_values is not None:
        class_names = list(class_values)
        word_to_idx = {cls: i for i, cls in enumerate(class_values)}
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)
    else:
        class_names = [str(i) for i in range(num_classes)]
        yte_idx = torch.tensor(yte, dtype=torch.long)

    L_in = Xtr.shape[-1]
    seq_len = max(21, Xte.shape[1])
    args.feats_size = L_in

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
    if args.model == 'TimeMIL':
        milnet = TimeMIL(args.feats_size, mDim=args.embed,
                         n_classes=num_classes,
                         dropout=args.dropout_node if hasattr(args, 'dropout_node') else 0.0,
                         max_seq_len=seq_len, is_instance=True).to(device)
    elif args.model == 'newTimeMIL':
        milnet = newTimeMIL(args.feats_size, mDim=args.embed,
                            n_classes=num_classes,
                            dropout=args.dropout_node if hasattr(args, 'dropout_node') else 0.0,
                            max_seq_len=seq_len, is_instance=True).to(device)
    elif args.model == 'AmbiguousMIL':
        milnet = AmbiguousMILwithCL(args.feats_size, mDim=args.embed,
                                    n_classes=num_classes,
                                    dropout=args.dropout_node if hasattr(args, 'dropout_node') else 0.0, is_instance=True).to(device)
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
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='newTimeMIL', type=str, help='MIL model: TimeMIL | newTimeMIL | AmbiguousMIL')
    parser.add_argument('--model_path', default='./savemodel/InceptBackbone/BasicMotions/exp_7/weights/best_newTimeMIL.pth',
                        type=str, help='Target model path for evaluation')
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--num_random', default=3, type=int, help='Random baseline repetition for AOPCR')
    parser.add_argument('--dropout_node', default=0.0, type=float, help='Dropout rate for classifier heads')
    parser.add_argument('--no_fallback_to_predicted', action='store_false', dest='fallback_to_predicted',
                        help='If set, skip bags with no positive labels instead of using argmax prediction')
    parser.add_argument('--cls_threshold', default=0.5, type=float, help='Threshold for multi-label classification metrics')
    parser.add_argument('--datatype', default="mixed", type=str, help='Choose datatype between original and mixed')

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

    # DataLoader
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
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

    # AOPCR
    print("Computing class-wise AOPCR ...")
    aopcr_c, aopcr_w_avg, aopcr_mean, aopcr_sum, M_expl, M_rand, alphas, counts = compute_classwise_aopcr(
        milnet,
        testloader,
        args,
        stop=0.5,
        step=0.05,
        n_random=args.num_random,
        pred_threshold=0.5
    )

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


if __name__ == '__main__':
    main()
