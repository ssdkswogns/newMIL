# viz_pred_inst_vs_gt.py
# -*- coding: utf-8 -*-
"""
사용 예:
python visualize.py \
  --dataset BasicMotions \
  --model newTimeMIL \
  --datatype mixed \
  --ckpt ./savemodel/InceptBackbone/BasicMotions/exp_6/weights/best_newTimeMIL.pth \
  --save_dir ./explain_pred \
  --batch_size 128 \
  --max_save 200

목표:
- mixed testset에서 y_inst(=GT per-timestep one-hot)과
- 모델로부터 얻은 pred_inst(=per-timestep class id)
를 나란히 밴드로 시각화하여 저장
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, TensorDataset

from aeon.datasets import load_classification
from syntheticdataset import MixedSyntheticBagsConcatK
from mydataload import loadorean

from models.timemil import TimeMIL, newTimeMIL
from models.expmil import AmbiguousMILwithCL


# -------------------------------
# util: attn에서 class->time 블록 추출
# -------------------------------
def attn_class_to_time(attn4d: torch.Tensor, C: int, T: int) -> torch.Tensor:
    """
    attn4d: [B, H, L_pad, L_pad]
    반환 :  [B, H, C, T]   (class queries -> time keys)
    규칙 : [pad | C(class tokens) | T(time tokens)]
    """
    if attn4d is None:
        raise ValueError("attn_layer2 is None")

    if attn4d.dim() != 4:
        raise ValueError(f"attn_layer2 must be 4D, got {tuple(attn4d.shape)}")

    B, H, Lq, Lk = attn4d.shape
    if Lq != Lk:
        raise ValueError(f"attn must be square on last two dims, got {Lq}x{Lk}")

    L_pad = Lq
    pad = L_pad - (C + T)
    if pad < 0:
        raise ValueError(f"Computed negative pad={pad}. Check C,T and attn shape={tuple(attn4d.shape)}")

    return attn4d[:, :, pad:pad + C, pad + C:pad + C + T]


# -------------------------------
# 데이터: 테스트셋 구성 (가능하면 y_inst 포함)
# -------------------------------
def build_testset(args):
    class_names = None

    # ---- NPZ 로더 계열 ----
    if args.dataset == 'PAMAP2':
        # loadorean이 return_instance_labels 지원한다고 가정 (old code 흐름)
        testset = loadorean(args, split='test', return_instance_labels=True)
        seq_len, num_classes, in_dim = testset.max_len, testset.num_class, testset.feat_in
        return testset, seq_len, num_classes, in_dim, class_names

    if args.dataset in ['JapaneseVowels', 'SpokenArabicDigits', 'CharacterTrajectories', 'InsectWingbeat']:
        testset = loadorean(args, split='test', return_instance_labels=True)
        seq_len, num_classes, in_dim = testset.max_len, testset.num_class, testset.feat_in
        return testset, seq_len, num_classes, in_dim, class_names

    # ---- AEON ----
    _, _, meta = load_classification(name=args.dataset, split='train', extract_path='./data')
    Xte, yte, _ = load_classification(name=args.dataset, split='test', extract_path='./data')

    class_values = meta.get('class_values', None)
    if class_values is not None:
        class_names = list(class_values)
        word_to_idx = {cls: i for i, cls in enumerate(class_values)}
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)
    else:
        yte_idx = torch.tensor(yte, dtype=torch.long)

    Xte = torch.from_numpy(Xte).permute(0, 2, 1).float()  # [N,T,D]
    num_classes = len(class_names) if class_names is not None else int(yte_idx.max().item() + 1)
    in_dim = Xte.shape[-1]
    seq_len = Xte.shape[1]

    # mixed 인스턴스 GT가 필요하므로 합성 bag 생성 (return_instance_labels=True)
    if args.datatype != "mixed":
        raise ValueError("이 스크립트는 GT 인스턴스 라벨(y_inst)이 필요하므로 --datatype mixed 를 권장합니다.")

    testset = MixedSyntheticBagsConcatK(
        X=Xte, y_idx=yte_idx, num_classes=num_classes,
        total_bags=len(Xte),
        seed=args.seed + 1,
        return_instance_labels=True
    )
    return testset, seq_len, num_classes, in_dim, class_names


# -------------------------------
# 모델 빌드 (학습 코드와 일치하도록)
# -------------------------------
def build_model(args, in_dim, num_classes, seq_len, device):
    if args.model == 'TimeMIL':
        m = TimeMIL(
            in_features=in_dim, n_classes=num_classes,
            mDim=args.embed, max_seq_len=seq_len,
            dropout=args.dropout_node, is_instance=True
        ).to(device)
    elif args.model == 'newTimeMIL':
        m = newTimeMIL(
            in_features=in_dim, n_classes=num_classes,
            mDim=args.embed, max_seq_len=seq_len,
            dropout=args.dropout_node, is_instance=True
        ).to(device)
    elif args.model == 'AmbiguousMIL':
        m = AmbiguousMILwithCL(
            in_dim, mDim=args.embed, n_classes=num_classes,
            dropout=args.dropout_node, is_instance=True
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return m


# -------------------------------
# 그림 저장: GT band / Pred band / (옵션) attn heatmap
# -------------------------------
def save_gt_pred_plot(
    y_inst_onehot: torch.Tensor,         # [T, C]
    pred_inst: torch.Tensor,             # [T] (class id)
    save_path: str,
    class_names=None,
    title: str = "",
    attn_ct: torch.Tensor = None         # [C_attn, T] optional
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_gt = torch.argmax(y_inst_onehot, dim=-1).detach().cpu().numpy()  # [T]
    pred = pred_inst.detach().cpu().numpy()  # [T]
    T = y_gt.shape[0]
    C = y_inst_onehot.shape[1]

    cmap_fixed = plt.get_cmap('tab20', C)
    norm_fixed = mcolors.Normalize(vmin=-0.5, vmax=C - 0.5)

    use_attn = attn_ct is not None
    rows = 3 if use_attn else 2
    fig_w = max(10.0, 0.05 * T + 8.0)
    fig_h = 7.0 if use_attn else 5.0

    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # (1) GT
    ax1 = plt.subplot(rows, 1, 1)
    im1 = ax1.imshow(y_gt[None, :], aspect="auto", interpolation="nearest", cmap=cmap_fixed, norm=norm_fixed)
    ax1.set_yticks([]); ax1.set_xticks([])
    ax1.set_title(f"GT inst label (argmax)   {title}".strip())
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)
    cbar1.set_label("class id")
    if class_names and len(class_names) <= 20:
        cbar1.set_ticks(np.arange(C))
        cbar1.set_ticklabels(class_names)

    # (2) Pred
    ax2 = plt.subplot(rows, 1, 2, sharex=ax1)
    im2 = ax2.imshow(pred[None, :], aspect="auto", interpolation="nearest", cmap=cmap_fixed, norm=norm_fixed)
    ax2.set_yticks([]); ax2.set_xticks([])
    ax2.set_title("Pred inst (per-time class id)")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
    cbar2.set_label("class id")

    # (3) Attn heatmap (optional)
    if use_attn:
        attn_np = attn_ct.detach().cpu().numpy() if isinstance(attn_ct, torch.Tensor) else np.asarray(attn_ct)
        ax3 = plt.subplot(rows, 1, 3)
        im3 = ax3.imshow(attn_np, aspect="auto", interpolation="nearest", cmap="viridis")
        ax3.set_title(f"Attention (class->time)  shape=({attn_np.shape[0]}x{attn_np.shape[1]})")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Class token")
        step = max(1, T // 20)
        ax3.set_xticks(np.arange(0, T, step))
        plt.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print("[Saved]", save_path)


# -------------------------------
# pred_inst 추출 (모델별)
# -------------------------------
@torch.no_grad()
def infer_pred_inst(args, model, x, y_inst_onehot):
    """
    반환:
      pred_inst: [B, T] class id
      attn_ct:   [B, C_attn, T] or None
    """
    device = x.device
    B, T, _ = x.shape
    C_gt = y_inst_onehot.shape[-1]

    if args.model == "AmbiguousMIL":
        # 학습 코드에서: bag_prediction, instance_pred, weighted_instance_pred, ...
        out = model(x)
        # out unpack (안전하게 길이 기반)
        if len(out) >= 3:
            # AmbiguousMILwithCL forward가
            # (bag_prediction, instance_pred, weighted_instance_pred, non_weighted..., x_cls, x_seq, attn1, attn2)
            # 형태인 것으로 가정
            bag_pred = out[0]
            instance_pred = out[1]
            weighted_instance_pred = out[2]
            attn_layer2 = out[-1] if len(out) >= 8 else None
        else:
            raise ValueError("AmbiguousMIL output format unexpected.")

        # pred_inst: weighted_instance_pred 기준
        pred_inst = torch.argmax(weighted_instance_pred, dim=2)  # [B,T]
        attn_ct = None
        return pred_inst, attn_ct

    # TimeMIL / newTimeMIL
    out = model(x)

    # old code처럼 tuple/list로 나오는 케이스 우선 처리
    attn_layer2 = None
    if isinstance(out, (tuple, list)):
        # newTimeMIL: (bag_logits, x_tokens, attn1, attn2) or similar
        # TimeMIL:    (bag_logits, x_tokens, attn2) or similar
        attn_layer2 = out[-1]
    else:
        # bag logits만 반환하면 pred_inst를 만들 근거가 부족
        raise ValueError("Model forward did not return attention. Ensure is_instance=True and forward returns attn_layer2.")

    # class token 수: newTimeMIL은 보통 C_gt, TimeMIL은 1(CLS)일 수 있으나
    # pred_inst를 만들려면 class-token이 C_gt여야 의미가 있으므로:
    # - TimeMIL이 실제로 1 CLS라면 pred_inst는 정의가 애매합니다.
    if args.model == "TimeMIL":
        # TimeMIL에서 attn 기반 pred_inst를 정의하고 싶으면
        # 구현/forward에서 class token을 C개로 두는지 먼저 확인 필요.
        # 여기서는 "가능하면 C_gt로 간주" 시도하되 실패 시 예외를 내도록 처리.
        C_tok = C_gt
    else:
        C_tok = C_gt

    attn_bhct = attn_class_to_time(attn_layer2, C=C_tok, T=T)  # [B,H,C,T]
    attn_ct = attn_bhct.mean(dim=1)  # [B,C,T]
    pred_inst = torch.argmax(attn_ct, dim=1)  # [B,T]
    return pred_inst, attn_ct


def main():
    parser = argparse.ArgumentParser(description="Qualitative viz: pred_inst vs GT (mixed testset)")

    parser.add_argument('--dataset', default="BasicMotions", type=str)
    parser.add_argument('--datatype', default="mixed", type=str, choices=["mixed"])
    parser.add_argument('--model', default='newTimeMIL', type=str, choices=['TimeMIL', 'newTimeMIL', 'AmbiguousMIL'])

    parser.add_argument('--embed', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--dropout_node', default=0.2, type=float)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', default='./explain_pred', type=str)
    parser.add_argument('--prepared_npz', type=str, default='./data/PAMAP2.npz')

    parser.add_argument('--max_save', default=200, type=int, help='최대 저장 샘플 수 (전체 저장 방지)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # testset
    testset, seq_len, num_classes, in_dim, class_names = build_testset(args)

    out_dir = os.path.join(args.save_dir, args.dataset, args.model)
    os.makedirs(out_dir, exist_ok=True)

    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False
    )

    model = build_model(args, in_dim, num_classes, seq_len, device)
    sd = torch.load(args.ckpt, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        model.load_state_dict(sd, strict=False)
    model.eval()

    saved = 0
    global_idx = 0

    with torch.no_grad():
        for batch in testloader:
            # mixed testset: (bags, y_bags, y_insts)
            if len(batch) == 3:
                bags, y_bag, y_inst = batch
            else:
                raise ValueError("Expected (bags, y_bag, y_inst). Ensure return_instance_labels=True.")

            x = bags.to(device)                 # [B,T,D]
            y_inst = y_inst.to(device)          # [B,T,C]

            pred_inst_b, attn_ct_b = infer_pred_inst(args, model, x, y_inst)  # [B,T], [B,C,T] or None

            B = x.size(0)
            for i in range(B):
                if saved >= args.max_save:
                    print(f"[Done] Reached max_save={args.max_save}")
                    return

                yi = y_inst[i]                 # [T,C]
                pi = pred_inst_b[i]            # [T]
                attn_i = attn_ct_b[i] if attn_ct_b is not None else None

                save_path = os.path.join(out_dir, f"idx{global_idx:05d}_gt_vs_pred.png")
                title = f"{args.dataset} | {args.model} | idx={global_idx}"
                save_gt_pred_plot(
                    y_inst_onehot=yi,
                    pred_inst=pi,
                    save_path=save_path,
                    class_names=class_names,
                    title=title,
                    attn_ct=attn_i
                )
                global_idx += 1
                saved += 1


if __name__ == "__main__":
    main()
