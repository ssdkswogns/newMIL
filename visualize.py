# visualize.py
# -*- coding: utf-8 -*-
"""
사용 예:
python visualize.py \
  --dataset BasicMotions \
  --model newTimeMIL \
  --ckpt ./savemodel/InceptBackbone/BasicMotions/exp_6/weights/best_newTimeMIL.pth \
  --save_dir ./explain
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader

# 프로젝트 모듈
from models.timemil import TimeMIL, newTimeMIL, AmbiguousMIL
from aeon.datasets import load_classification
from syntheticdataset import MixedSyntheticBags
from mydataload import loadorean

# -------------------------------
# 데이터: 테스트셋만 구성 (y_inst 포함)
# -------------------------------
def build_testset(args):
    class_names = None

    # ---- NPZ가 들어온 경우: loadorean + VizWrapper ----
    if args.dataset == 'PAMAP2':
        testset = loadorean(args, split='test', return_instance_labels=True)
        seq_len, num_classes, in_dim = testset.max_len, testset.num_class, testset.feat_in
        return testset, seq_len, num_classes, in_dim, class_names

    # ---- 기존 AEON 분기 ----
    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:
        testset = loadorean(args, split='test')
        seq_len, num_classes, in_dim = testset.max_len, testset.num_class, testset.feat_in
        return testset, seq_len, num_classes, in_dim, class_names
    
    # ---- 기존 합성 분기 ----
    _, _, meta = load_classification(name=args.dataset, split='train')
    Xte, yte, _ = load_classification(name=args.dataset, split='test')

    class_values = meta.get('class_values', None)
    if class_values is not None:
        class_names = list(class_values)
        word_to_idx = {cls:i for i, cls in enumerate(class_values)}
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)
    else:
        yte_idx = torch.tensor(yte, dtype=torch.long)

    Xte = torch.from_numpy(Xte).permute(0, 2, 1).float()   # [N,T,D]
    num_classes = len(class_names) if class_names is not None else int(yte_idx.max().item()+1)
    in_dim = Xte.shape[-1]
    seq_len = Xte.shape[1]

    testset = MixedSyntheticBags(
        X=Xte, y_idx=yte_idx, num_classes=num_classes,
        total_bags=max(1000, len(Xte)),
        probs={'orig':0.4, 2:0.4, 3:0.2}, total_len=seq_len,
        min_seg_len=32, ensure_distinct_classes=True,
        seed=args.seed, return_instance_labels=True
    )
    return testset, seq_len, num_classes, in_dim, class_names

# -------------------------------
# 모델
# -------------------------------
def build_model(args, in_dim, num_classes, seq_len, device):
    if args.model == 'TimeMIL':
        m = TimeMIL(in_features=in_dim, n_classes=num_classes,
                    mDim=args.embed, max_seq_len=seq_len,
                    dropout=0.2, is_instance=True).to(device)
    elif args.model == 'newTimeMIL':
        m = newTimeMIL(in_features=in_dim, n_classes=num_classes,
                       mDim=args.embed, max_seq_len=seq_len,
                       dropout=0.2, is_instance=True).to(device)
    else:
        m = AmbiguousMIL(in_features=in_dim,mDim=args.embed,n_classes =num_classes,
                        dropout=0.2, max_seq_len = seq_len, is_instance=True).to(device)
    return m

# -------------------------------
# 한 장에: 위=inst argmax 띠(색상 고정),
#        중간=어텐션 argmax 띠(옵션; newTimeMIL에서만 의미 있음),
#        아래=어텐션 히트맵(C×T)  ※ TimeMIL이면 C=1
# -------------------------------
def visualize_inst_and_attn(attn_ct, y_inst, save_path, class_names=None, title="", pred_band=None):
    """
    attn_ct: [C_attn, T]  (헤드 평균 및 필요한 슬라이스 적용 후의 attention)
    y_inst : [T, C_gt]    (one-hot)
    pred_band: [T]        (optional) 어텐션 argmax 결과; TimeMIL이면 None 권장
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1) GT inst argmax (1×T)
    y_argmax = torch.argmax(y_inst, dim=-1)  # [T]
    y = y_argmax.detach().cpu().numpy()
    T = y.shape[0]
    C_gt = y_inst.shape[1]  # 색상 매핑 고정용 (데이터셋 클래스 수)

    # 2) 어텐션 히트맵 입력 정리
    if isinstance(attn_ct, torch.Tensor):
        attn_ct_np = attn_ct.detach().cpu().numpy()
    else:
        attn_ct_np = np.asarray(attn_ct)
    C_attn, T_attn = attn_ct_np.shape
    assert T_attn == T, f"Time length mismatch: attention T={T_attn}, labels T={T}"

    # Figure 설정
    fig_w = max(8.0, 0.04*T + 6.0)
    use_pred_band = (pred_band is not None)
    rows = 3 if use_pred_band else 2
    fig_h = 8.0 if use_pred_band else 6.2
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # ---- 색상 고정(두 띠 동일 팔레트) ----
    cmap_fixed = plt.get_cmap('tab20', C_gt)
    norm_fixed = mcolors.Normalize(vmin=-0.5, vmax=C_gt - 0.5)

    # (1) 위 패널: inst argmax 띠
    ax1 = plt.subplot(rows, 1, 1)
    band_gt = y[None, :]  # [1,T]
    im1 = ax1.imshow(band_gt, aspect='auto', interpolation='nearest',
                     cmap=cmap_fixed, norm=norm_fixed)
    ax1.set_yticks([]); ax1.set_xticks([])
    ttl = "Per-time inst label argmax (1×T)"
    if title: ttl += "  |  " + title
    ax1.set_title(ttl)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)
    cbar1.set_label("class id")
    if class_names and len(class_names) <= 20:
        cbar1.set_ticks(np.arange(C_gt))
        cbar1.set_ticklabels(class_names)

    # (2) 중간 패널: 어텐션 argmax 띠 (newTimeMIL에서만; TimeMIL은 None)
    if use_pred_band:
        axm = plt.subplot(rows, 1, 2, sharex=ax1)
        pb = pred_band.detach().cpu().numpy() if isinstance(pred_band, torch.Tensor) else np.asarray(pred_band)
        band_pred = pb[None, :]  # [1,T]
        im_mid = axm.imshow(band_pred, aspect='auto', interpolation='nearest',
                            cmap=cmap_fixed, norm=norm_fixed)
        axm.set_yticks([]); axm.set_xticks([])
        axm.set_title("Per-time attention argmax (1×T)")
        cbar_mid = plt.colorbar(im_mid, ax=axm, fraction=0.03, pad=0.02)
        cbar_mid.set_label("class id (pred)")

    # (마지막) 아래 패널: 어텐션 히트맵 (C_attn×T)
    ax_last = plt.subplot(rows, 1, rows)
    im2 = ax_last.imshow(attn_ct_np, aspect='auto', interpolation='nearest', cmap='viridis')
    ax_last.set_title(f"Attention map (class tokens → time)  shape=({C_attn}×{T})")
    ax_last.set_xlabel("Time (T)")
    ax_last.set_ylabel("Class token" if C_attn > 1 else "CLS")
    if C_attn <= 50:
        if class_names and C_attn == len(class_names):
            ax_last.set_yticks(np.arange(C_attn)); ax_last.set_yticklabels(class_names)
        else:
            ax_last.set_yticks(np.arange(C_attn)); ax_last.set_yticklabels([f"class-{i}" for i in range(C_attn)] if C_attn > 1 else ["CLS"])
    else:
        ax_last.set_yticks([])
    step = max(1, T // 20)
    ax_last.set_xticks(np.arange(0, T, step))
    cbar2 = plt.colorbar(im2, ax=ax_last, fraction=0.03, pad=0.02)
    cbar2.set_label("attention")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("[Saved]", save_path)

# === 헬퍼: 클래스→시간 블록 추출 ===
def attn_class_to_time(attn4d: torch.Tensor, C: int, T: int) -> torch.Tensor:
    """
    attn4d: [B, H, L_pad, L_pad]  (NystromAttention에서 반환)
    반환:   [B, H, C, T]          (클래스 토큰 질의 × 시간 토큰 키)
    규칙:   시퀀스는 [pad | C(class) | T(time)], pad = L_pad - (C+T)
    """
    if attn4d.dim() != 4:
        raise ValueError(f"attn_layer2 must be 4D, got {tuple(attn4d.shape)}")
    B, H, Lq, Lk = attn4d.shape
    if Lq != Lk:
        raise ValueError(f"attn must be square on last two dims, got {Lq}x{Lk}")
    L_pad = Lq
    pad = L_pad - (C + T)
    if pad < 0:
        raise ValueError(f"Computed negative pad={pad}. Check C,T and attn shape.")
    # 질의: 클래스 토큰 영역, 키: 시간 토큰 영역
    return attn4d[:, :, pad : pad + C, pad + C : pad + C + T]

# -------------------------------
# 메인
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Instance argmax + (pred) attention argmax + Attention heatmap (testset only)")
    parser.add_argument('--dataset', default="BasicMotions", type=str)
    parser.add_argument('--model', default='newTimeMIL', type=str, choices=['TimeMIL','newTimeMIL','AmbiguousMIL'])
    parser.add_argument('--embed', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--ckpt', default='', type=str, required=True)
    parser.add_argument('--save_dir', default='./explain', type=str)
    parser.add_argument('--prepared_npz', type=str, default='./data/PAMAP2.npz', help='npz path for PAMAP2)')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 테스트셋
    testset, seq_len, num_classes, in_dim, class_names = build_testset(args)

    # 저장 폴더: save_dir/<dataset>/<model>/
    out_dir = os.path.join(args.save_dir, args.dataset, args.model)
    os.makedirs(out_dir, exist_ok=True)

    # 큰 배치로 전체 순회
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False
    )

    # 모델
    model = build_model(args, in_dim, num_classes, seq_len, device)
    sd = torch.load(args.ckpt, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        model.load_state_dict(sd, strict=False)
    model.eval()

    # 전체 배치 순회 저장
    global_idx = 0
    with torch.no_grad():
        for bags, y_bags, y_insts in testloader:
            # 입력: [B,T,D], [B,C_gt], [B,T,C_gt]
            x = bags.to(device)
            if args.model == 'AmbiguousMIL':
                logits, instance_logits, x_cls, x_seq, attn_layer1, attn_layer2 = model(x)
            else:
                logits, x_tokens, attn_layer1, attn_layer2 = model(x)

            B, T, C_gt = y_insts.shape

            # --- TimeMIL vs newTimeMIL: 클래스 토큰 개수 결정 ---
            if args.model == 'TimeMIL':
                C_tok = 1             # 클래스 토큰 1개(CLS)
                show_pred_band = False  # argmax 띠 무의미 → 비표시
            else:
                C_tok = C_gt          # 클래스 토큰 = 데이터셋 클래스 수
                show_pred_band = True
            
            # === attn_layer2에서 클래스→시간 블록 뽑기 ===
            attn_bhct = attn_class_to_time(attn_layer2, C=C_tok, T=T)  # [B,H,C_tok,T]
            attn_mean = attn_bhct.mean(dim=1)                           # [B,C_tok,T]

            if args.model == 'AmbiguousMIL':
                pred_bands = torch.argmax(instance_logits, dim=2) if show_pred_band else None  # [B,T]
            else:
                # pred_band: newTimeMIL에서만 계산
                # eps = 1e-9
                # # attn_mean: [B, C_tok, T]
                # attn_colsums = attn_mean.sum(dim=1, keepdim=True).clamp_min(eps)   # [B,1,T]
                # attn_norm = attn_mean / attn_colsums                               # [B,C_tok,T]  # 시간 t별 클래스 분율
                pred_bands = torch.argmax(attn_mean, dim=1) if show_pred_band else None  # [B,T]

            for i in range(x.shape[0]):
                yi = y_insts[i]                      # [T,C_gt]
                # attn_ct_i = attn_mean[i]             # [C_tok,T]
                attn_ct_i = attn_mean[i]
                pred_band_i = pred_bands[i] if show_pred_band else None  # [T] or None

                save_path = os.path.join(out_dir, f"{args.model}_{args.dataset}_idx{global_idx:05d}_inst_plus_attn.png")
                title = f"{args.dataset} | {args.model} | idx={global_idx}"
                visualize_inst_and_attn(attn_ct_i, yi, save_path,
                                        class_names=class_names,
                                        title=title,
                                        pred_band=pred_band_i)
                global_idx += 1

if __name__ == "__main__":
    main()
