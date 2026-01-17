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

from models.timemil_old import TimeMIL, newTimeMIL
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
        x_seq: torch.Tensor,               # [T, D]  <-- 추가
        y_inst_onehot: torch.Tensor,       # [T, C]
        pred_inst: torch.Tensor,           # [T]
        save_path: str,
        class_names=None,
        title: str = "",
        attn_ct: torch.Tensor = None,      # [C_attn, T] optional
        feat_names=None,                  # optional: 길이 D의 리스트
        plot_k: int = 6,
        plot_idx=None,                    # optional: feature index list
        plot_norm: str = "z",             # none|z|minmax
        plot_offset: float = 2.5,
        plot_alpha: float = 0.9,
    ):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # --- numpy로 변환 ---
        x_np = x_seq.detach().cpu().numpy()          # [T,D]
        y_gt = torch.argmax(y_inst_onehot, dim=-1).detach().cpu().numpy()  # [T]
        pred = pred_inst.detach().cpu().numpy()      # [T]
        T = y_gt.shape[0]
        C = y_inst_onehot.shape[1]
        D = x_np.shape[1]

        # --- 어떤 feature를 그릴지 선택 ---
        if plot_idx is not None and len(plot_idx) > 0:
            sel = [i for i in plot_idx if 0 <= i < D]
            if len(sel) == 0:
                sel = list(range(min(D, plot_k))) if plot_k > 0 else list(range(D))
        else:
            if plot_k == 0 or plot_k >= D:
                sel = list(range(D))
            else:
                # 분산 큰 순서대로 상위 K개 선택 (정성 시각화에 보통 유용)
                v = np.var(x_np, axis=0)
                sel = np.argsort(-v)[:plot_k].tolist()

        x_sel = x_np[:, sel]  # [T,K]
        K = x_sel.shape[1]

        # --- 정규화 ---
        if plot_norm == "z":
            mu = x_sel.mean(axis=0, keepdims=True)
            sd = x_sel.std(axis=0, keepdims=True) + 1e-9
            x_sel = (x_sel - mu) / sd
        elif plot_norm == "minmax":
            mn = x_sel.min(axis=0, keepdims=True)
            mx = x_sel.max(axis=0, keepdims=True)
            x_sel = (x_sel - mn) / (mx - mn + 1e-9)
        elif plot_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown plot_norm={plot_norm}")

        # --- 밴드 컬러맵 ---
        cmap_fixed = plt.get_cmap('tab20', C)
        norm_fixed = mcolors.Normalize(vmin=-0.5, vmax=C - 0.5)

        use_attn = attn_ct is not None
        rows = 4 if use_attn else 3

        fig_w = max(12.0, 0.05 * T + 9.0)
        fig_h = 10.0 if use_attn else 8.0
        plt.figure(figsize=(fig_w, fig_h), dpi=150)

        # (1) Input sequence plot
        ax0 = plt.subplot(rows, 1, 1)
        t = np.arange(T)

        # offset stacked plot
        for j in range(K):
            y = x_sel[:, j] + j * plot_offset
            ax0.plot(t, y, linewidth=1.0, alpha=plot_alpha)

        ax0.set_title(f"Input sequence  (showing {K}/{D} features)   {title}".strip())
        ax0.set_xlim(0, T - 1)
        ax0.set_yticks([j * plot_offset for j in range(K)])

        # ytick label: feature index or name
        if feat_names is not None and len(feat_names) == D:
            ylabels = [feat_names[i] for i in sel]
        else:
            ylabels = [f"f{idx}" for idx in sel]
        ax0.set_yticklabels(ylabels)

        ax0.set_xlabel("Time")
        ax0.grid(True, axis="x", linewidth=0.3, alpha=0.4)

        # (2) GT band
        ax1 = plt.subplot(rows, 1, 2, sharex=ax0)
        im1 = ax1.imshow(y_gt[None, :], aspect="auto", interpolation="nearest", cmap=cmap_fixed, norm=norm_fixed)
        ax1.set_yticks([]); ax1.set_xticks([])
        ax1.set_title("GT inst label (argmax)")
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)
        cbar1.set_label("class id")
        if class_names and len(class_names) <= 20:
            cbar1.set_ticks(np.arange(C))
            cbar1.set_ticklabels(class_names)

        # (3) Pred band
        ax2 = plt.subplot(rows, 1, 3, sharex=ax0)
        im2 = ax2.imshow(pred[None, :], aspect="auto", interpolation="nearest", cmap=cmap_fixed, norm=norm_fixed)
        ax2.set_yticks([]); ax2.set_xticks([])
        ax2.set_title("Pred inst (per-time class id)")
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
        cbar2.set_label("class id")

        # (4) Attn heatmap
        if use_attn:
            attn_np = attn_ct.detach().cpu().numpy() if isinstance(attn_ct, torch.Tensor) else np.asarray(attn_ct)
            ax3 = plt.subplot(rows, 1, 4, sharex=ax0)
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
    parser.add_argument('--plot_k', default=0, type=int,
                    help='input feature 중 상위 K개만 시각화 (0이면 모두)')
    parser.add_argument('--plot_idx', default=None, type=str,
                        help='특정 feature index만 시각화. 예: "0,2,5" (지정 시 plot_k 무시)')
    parser.add_argument('--plot_norm', default='none', type=str, choices=['none', 'z', 'minmax'],
                        help='input feature 정규화 방식')
    parser.add_argument('--plot_offset', default=2.5, type=float,
                        help='멀티채널을 한 축에 겹쳐 그릴 때 채널 간 vertical offset')
    parser.add_argument('--plot_alpha', default=0.9, type=float,
                        help='input plot 선 투명도')
    args = parser.parse_args()

    def parse_plot_idx(s):
        if s is None:
            return None
        s = s.strip()
        if not s:
            return None
        return [int(x) for x in s.split(',')]

    args.plot_idx = parse_plot_idx(args.plot_idx)

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
                save_path = os.path.join(out_dir, f"idx{global_idx:05d}_gt_vs_pred.png")
                title = f"{args.dataset} | {args.model} | idx={global_idx}"

                xi = x[i].detach()             # [T,D]
                yi = y_inst[i]                 # [T,C]
                pi = pred_inst_b[i]            # [T]
                attn_i = attn_ct_b[i] if attn_ct_b is not None else None

                save_gt_pred_plot(
                    x_seq=xi,
                    y_inst_onehot=yi,
                    pred_inst=pi,
                    save_path=save_path,
                    class_names=class_names,
                    title=title,
                    attn_ct=attn_i,
                    feat_names=None,                 # 필요하면 여기에 feature name list
                    plot_k=args.plot_k,
                    plot_idx=args.plot_idx,
                    plot_norm=args.plot_norm,
                    plot_offset=args.plot_offset,
                    plot_alpha=args.plot_alpha,
                )
                saved += 1
                global_idx += 1


if __name__ == "__main__":
    main()
