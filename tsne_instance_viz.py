# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from aeon.datasets import load_classification
from mydataload import loadorean
from dba_dataloader import build_dba_for_timemil, build_dba_windows_for_mixed
from syntheticdataset import MixedSyntheticBagsConcatK
from models.timemil import AmbiguousMILwithCL


def build_testset_and_args(args):
    """
    main()의 데이터 구성 부분을 '테스트셋만' 간단히 재현한 함수입니다.
    - seq_len, num_classes, feats_size를 셋업하고
    - testset 및 testloader를 반환합니다.
    """
    if args.dataset in ['JapaneseVowels', 'SpokenArabicDigits',
                        'CharacterTrajectories', 'InsectWingbeat']:
        trainset = loadorean(args, split='train')
        testset  = loadorean(args, split='test')

        seq_len     = trainset.max_len
        num_classes = trainset.num_class
        L_in        = trainset.feat_in

        args.seq_len    = seq_len
        args.feats_size = L_in
        args.num_classes = num_classes

    elif args.dataset in ['PAMAP2']:
        trainset = loadorean(args, split='train')
        testset  = loadorean(args, split='test')

        seq_len     = trainset.max_len
        num_classes = trainset.num_class
        L_in        = trainset.feat_in

        args.seq_len    = seq_len
        args.feats_size = L_in
        args.num_classes = num_classes

    elif args.dataset == 'dba':
        if args.datatype == 'original':
            # window 하나가 bag
            trainset, testset, seq_len, num_classes, L_in = build_dba_for_timemil(args)
        elif args.datatype == 'mixed':
            # window 단위로 잘라서 concat-bag
            Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, L_in = build_dba_windows_for_mixed(args)

            # trainset/testset 구성은 main과 동일하게
            trainset = MixedSyntheticBagsConcatK(
                X=Xtr,
                y_idx=ytr_idx,
                num_classes=num_classes,
                total_bags=len(Xtr),
                concat_k=2,
                seed=args.seed,
            )
            testset = MixedSyntheticBagsConcatK(
                X=Xte,
                y_idx=yte_idx,
                num_classes=num_classes,
                total_bags=len(Xte),
                concat_k=2,
                seed=args.seed + 1,
                return_instance_labels=True,    # eval에서는 instance label도 있으면 좋음
            )
        else:
            raise ValueError(f"Unsupported datatype '{args.datatype}' for DBA dataset")

        args.seq_len    = seq_len
        args.feats_size = L_in
        args.num_classes = num_classes

    else:
        # UCR 계열 (main 코드의 else 부분 요약)
        Xtr, ytr, meta = load_classification(name=args.dataset, split='train', extract_path='./data')
        Xte, yte, _   = load_classification(name=args.dataset, split='test', extract_path='./data')

        word_to_idx = {cls: i for i, cls in enumerate(meta['class_values'])}
        ytr_idx = torch.tensor([word_to_idx[i] for i in ytr], dtype=torch.long)
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)

        Xtr = torch.from_numpy(Xtr).permute(0, 2, 1).float()
        Xte = torch.from_numpy(Xte).permute(0, 2, 1).float()

        num_classes = len(meta['class_values'])
        L_in = Xtr.shape[-1]
        seq_len = max(21, Xte.shape[1])

        args.num_classes = num_classes
        args.seq_len     = seq_len
        args.feats_size  = L_in

        if args.datatype == 'mixed':
            trainset = MixedSyntheticBagsConcatK(
                X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
                total_bags=len(Xtr),
                seed=args.seed
            )
            testset = MixedSyntheticBagsConcatK(
                X=Xte, y_idx=yte_idx, num_classes=num_classes,
                total_bags=len(Xte),
                seed=args.seed + 1,
                return_instance_labels=True
            )
        elif args.datatype == 'original':
            trainset = torch.utils.data.TensorDataset(
                Xtr, F.one_hot(ytr_idx, num_classes=num_classes).float()
            )
            testset = torch.utils.data.TensorDataset(
                Xte, F.one_hot(yte_idx, num_classes=num_classes).float()
            )
        else:
            raise ValueError(f"Unsupported datatype '{args.datatype}'")

    # testloader: batch_size=1 (bag 단위로 수집)
    testloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return testset, testloader, args


def collect_instance_embeddings(args, model, testloader, device,
                                max_instances=20000):
    """
    AmbiguousMILwithCL 모델에서
    - x_seq: [B, T, D]  (instance feature, 마지막 레이어 시퀀스 embedding)
    - instance_pred: [B, T, C] (instance-level logits)
    - bag_label: [B, C]

    을 모아서 t-SNE에 쓸 수 있도록 평탄화하여 반환합니다.

    max_instances: t-SNE 복잡도 때문에 너무 많을 경우 subsample.
    """
    model.eval()
    all_feats = []
    all_bag_labels = []
    all_inst_logits = []

    with torch.no_grad():
        for batch in testloader:
            if args.datatype == 'mixed':
                feats, bag_label, *_ = batch  # (feats, label, [y_inst])
            else:
                feats, bag_label = batch

            feats = feats.to(device)          # [B, T, F]
            bag_label = bag_label.to(device)  # [B, C]

            # AmbiguousMILwithCL forward:
            # bag_prediction, instance_pred, x_cls, x_seq, c_seq, attn1, attn2
            out = model(feats)
            bag_pred, instance_pred, x_cls, x_seq, c_seq, attn1, attn2 = out

            # x_seq: [B, T, D]
            B, T, D = x_seq.shape
            _, C = bag_label.shape

            # flatten: [B*T, D]
            feats_bt = x_seq.reshape(B * T, D)           # instance feature
            logits_bt = instance_pred.reshape(B * T, C)  # instance-level logits

            # bag label을 timestep마다 복제: [B*T, C]
            bag_label_bt = bag_label.unsqueeze(1).repeat(1, T, 1).reshape(B * T, C)

            all_feats.append(feats_bt.cpu())
            all_bag_labels.append(bag_label_bt.cpu())
            all_inst_logits.append(logits_bt.cpu())

            # 너무 많으면 중간에 끊기
            cur_N = sum(x.shape[0] for x in all_feats)
            if cur_N >= max_instances:
                break

    feats_cat = torch.cat(all_feats, dim=0).numpy()          # [N, D]
    bag_labels_cat = torch.cat(all_bag_labels, dim=0).numpy()  # [N, C]
    inst_logits_cat = torch.cat(all_inst_logits, dim=0).numpy()  # [N, C]

    return feats_cat, bag_labels_cat, inst_logits_cat


def run_tsne(features, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=200,
        n_iter=1000,
        metric='euclidean',
        random_state=random_state,
        init='pca',
    )
    emb = tsne.fit_transform(features)
    return emb  # [N, 2]


def plot_tsne(emb2d,
              bag_labels,
              inst_logits,
              save_prefix="tsne_ambiguousmil"):
    """
    emb2d      : [N, 2]
    bag_labels : [N, C] (one-hot/multi-hot)
    inst_logits: [N, C] (logits)

    - plot 1: Bag GT 기준 색칠
    - plot 2: Instance prediction 기준 색칠
    """
    N, _ = emb2d.shape
    C = bag_labels.shape[1]

    # ----- Bag label 기반 class index -----
    # multi-hot일 수도 있지만, 일단 argmax로 대표 class를 선정
    bag_cls = bag_labels.argmax(axis=1)  # [N]
    print(bag_cls)

    # ----- Instance-level prediction 기반 class index -----
    inst_prob = 1 / (1 + np.exp(-inst_logits))  # sigmoid
    inst_cls = inst_prob.argmax(axis=1)         # [N]

    # 공통 plotting 설정
    def _scatter_common(ax, labels, title):
        num_classes = 3
        from matplotlib.colors import ListedColormap, BoundaryNorm
        base = plt.get_cmap("tab10")
        cmap = ListedColormap(base.colors[:num_classes])
        norm = BoundaryNorm(range(num_classes+1), cmap.N)

        sc = ax.scatter(
            emb2d[:, 0], emb2d[:, 1],
            c=labels,        # ✅ labels 사용
            cmap=cmap,
            norm=norm,
            s=3,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_ticks(range(num_classes))

    # ----- Figure 구성 -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _scatter_common(axes[0], bag_cls, "t-SNE of instance features (colored by BAG GT)")
    _scatter_common(axes[1], inst_cls, "t-SNE of instance features (colored by INSTANCE prediction)")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    out_path = save_prefix + ".png"
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved t-SNE comparison figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization for AmbiguousMIL instance features")
    parser.add_argument('--dataset', default="dba", type=str)
    parser.add_argument('--datatype', default="mixed", type=str, help='original or mixed')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--embed', default=128, type=int)
    parser.add_argument('--dropout_node', default=0.2, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dba_root', type=str, default='./data/dba_data')
    parser.add_argument('--dba_window', type=int, default=50)
    parser.add_argument('--dba_stride', type=int, default=10)
    parser.add_argument('--dba_test_ratio', type=float, default=0.2)

    parser.add_argument('--ckpt', type=str, required=True,
                        help='path to best_AmbiguousMIL checkpoint (.pth)')
    parser.add_argument('--max_instances', type=int, default=20000,
                        help='max number of instances used for t-SNE')
    parser.add_argument('--save_prefix', type=str, default='./tsne/ambiguousmil_instance',
                        help='prefix for saved figure')

    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터셋/로더 구성
    _, testloader, args = build_testset_and_args(args)

    # 2) AmbiguousMILwithCL 모델 생성 (is_instance=True)
    model = AmbiguousMILwithCL(
        in_features=args.feats_size,
        mDim=args.embed,
        n_classes=args.num_classes,
        dropout=args.dropout_node,
        max_seq_len=args.seq_len,
        is_instance=True
    ).to(device)

    # 3) 체크포인트 로드
    print(f"[INFO] Loading checkpoint from: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    # 4) instance feature + label 수집
    feats, bag_labels, inst_logits = collect_instance_embeddings(
        args, model, testloader, device, max_instances=args.max_instances
    )
    print(f"[INFO] Collected instance embeddings: {feats.shape[0]} instances, dim={feats.shape[1]}")

    # 5) t-SNE 계산 (x_seq: 모델 제일 끝단 instance feature 사용)
    emb2d = run_tsne(feats, n_components=2, perplexity=30, random_state=args.seed)

    # 6) Bag label / Instance prediction 두 가지 기준으로 시각화
    plot_tsne(
        emb2d,
        bag_labels=bag_labels,
        inst_logits=inst_logits,
        save_prefix=args.save_prefix
    )


if __name__ == "__main__":
    main()
