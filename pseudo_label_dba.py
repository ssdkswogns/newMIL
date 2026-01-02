# pseudo_label_dba.py
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dba_dataloader import dba_scan_sequences, DBA_FEATURE_COLS, DBA_STYLE_TO_LABEL
# from models.timemil import AmbiguousMILwithCL
from models.expmil import AmbiguousMILwithCL
from models.milet import MILLET


def load_ambiguous_mil(
    ckpt_path: str,
    feats_size: int,
    num_classes: int,
    seq_len: int,
    embed_dim: int,
    dropout: float,
    device: torch.device,
):
    """
    학습된 AmbiguousMILwithCL 모델 로드 (is_instance=True).
    main_cl_fix.py에서 eval할 때 쓰신 생성자와 동일한 형태로 맞춤.
    """
    model = AmbiguousMILwithCL(
        feats_size,              # in_features
        mDim=embed_dim,
        n_classes=num_classes,
        dropout=dropout,
        is_instance=True,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_millet(
    ckpt_path: str,
    feats_size: int,
    num_classes: int,
    seq_len: int,
    embed_dim: int,
    dropout: float,
    device: torch.device,
):
    """
    학습된 AmbiguousMILwithCL 모델 로드 (is_instance=True).
    main_cl_fix.py에서 eval할 때 쓰신 생성자와 동일한 형태로 맞춤.
    """
    model = MILLET(
        feats_size,              # in_features
        mDim=embed_dim,
        n_classes=num_classes,
        dropout=dropout,
        max_seq_len=seq_len,
        is_instance=True,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def pseudo_label_one_sequence(
    args,
    model,
    df: pd.DataFrame,
    feature_cols,
    window_size: int,
    stride: int,
    num_classes: int,
    device: torch.device,
):
    """
    단일 시퀀스(원본 parsed_50hz.csv 전체)에 대해
    원본 길이 T 그대로 시점 단위 pseudo prob/label을 계산.

    return:
      prob  : [T, C] float32
      label : [T]   int64
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)  # [T, D]
    T, D = X.shape

    prob_sum = np.zeros((T, num_classes), dtype=np.float32)
    count    = np.zeros((T,), dtype=np.int32)

    if T == 0:
        raise RuntimeError("Empty sequence encountered.")

    if T <= window_size:
        # 너무 짧은 경우: 한 번만 통째로 넣음 (padding이 필요하면 여기서 추가)
        x_tensor = torch.from_numpy(X).unsqueeze(0).to(device)  # [1, T, D]
        if args.model == 'AmbiguousMIL':
            bag_prediction, instance_pred, weighted_instance_pred, non_weighted_instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = model(x_tensor)
            p_inst = torch.sigmoid(weighted_instance_pred)[0].cpu().numpy()   # [T, C]
        elif args.model == 'MILLET':
            bag_prediction, non_weighted_instance_pred, instance_pred = model(x_tensor)
            instance_pred = instance_pred.transpose(1,2)
            p_inst = torch.sigmoid(instance_pred)[0].cpu().numpy()   # [T, C]        

        prob_sum[:T] += p_inst
        count[:T] += 1
    else:
        # 1) 기본 stride대로 시작점 생성
        starts = list(range(0, T - window_size + 1, stride))
        # 2) 마지막 구간까지 모두 덮도록, 필요하면 마지막 start=T-window_size를 추가
        last_start = T - window_size
        if starts[-1] != last_start:
            starts.append(last_start)

        for start in starts:
            end = start + window_size
            x_win = X[start:end]  # [L, D]
            x_tensor = torch.from_numpy(x_win).unsqueeze(0).to(device)  # [1, L, D]
            if args.model == 'AmbiguousMIL':
                bag_prediction, instance_pred, weighted_instance_pred, non_weighted_instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = model(x_tensor)
                p_inst = torch.sigmoid(weighted_instance_pred)[0].cpu().numpy()  # [L, C]
            elif args.model == 'MILLET':
                bag_prediction, non_weighted_instance_pred, instance_pred = model(x_tensor)
                instance_pred = instance_pred.transpose(1,2)
                p_inst = torch.sigmoid(instance_pred)[0].cpu().numpy()  # [L, C]
            

            prob_sum[start:end] += p_inst
            count[start:end] += 1

    # 평균 확률
    prob = np.zeros_like(prob_sum)
    mask = count > 0
    prob[mask] = prob_sum[mask] / count[mask, None]
    # 혹시 count==0인 시점이 있으면 그대로 0 → argmax하면 0번 클래스가 됨 (필요시 별도 처리 가능)

    label = prob.argmax(axis=1).astype(np.int64)
    return prob, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dba_root', type=str, required=True,
                        help='1_xxx/aggressive/... 구조의 root 디렉토리')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='학습된 best_AmbiguousMIL.pth 경로')
    parser.add_argument('--out_root', type=str, required=True,
                        help='pseudo label 저장 루트 디렉토리')
    parser.add_argument('--window_size', type=int, required=True,
                        help='학습에 사용한 dba_window와 동일하게')
    parser.add_argument('--stride', type=int, required=True,
                        help='학습에 사용한 dba_stride와 동일하게')
    parser.add_argument('--embed', type=int, default=128,
                        help='학습 시 사용한 --embed 값')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='학습 시 사용한 --dropout_node 값')
    parser.add_argument('--gpu', type=int, default=0,
                        help='사용할 GPU index')
    parser.add_argument('--model', type=str, default='AmbiguousMIL',
                        help='사용할 model')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    num_classes = len(DBA_STYLE_TO_LABEL)
    feats_size  = len(DBA_FEATURE_COLS)   # feature dimension
    seq_len     = args.window_size       # max_seq_len은 학습 때 dba_window와 동일

    if args.model == 'AmbiguousMIL':
        model = load_ambiguous_mil(
            ckpt_path=args.ckpt,
            feats_size=feats_size,
            num_classes=num_classes,
            seq_len=seq_len,
            embed_dim=args.embed,
            dropout=args.dropout,
            device=device,
        )
    elif args.model == 'MILLET':
        model = load_millet(
            ckpt_path=args.ckpt,
            feats_size=feats_size,
            num_classes=num_classes,
            seq_len=seq_len,
            embed_dim=args.embed,
            dropout=args.dropout,
            device=device,
        )

    # 2) 시퀀스 스캔 (원본 CSV 경로 + style label)
    seq_list = dba_scan_sequences(args.dba_root)
    print(f"[Pseudo] total sequences: {len(seq_list)}")

    # class index -> name 매핑
    idx_to_style = {v: k for k, v in DBA_STYLE_TO_LABEL.items()}

    for csv_path, label_int in seq_list:
        print(f"[Pseudo] processing: {csv_path}")

        df = pd.read_csv(csv_path)

        prob, pseudo_label = pseudo_label_one_sequence(
            args=args,
            model=model,
            df=df,
            feature_cols=DBA_FEATURE_COLS,
            window_size=args.window_size,
            stride=args.stride,
            num_classes=num_classes,
            device=device,
        )

        # 원본 데이터 복사 + pseudo label 컬럼 추가
        df_out = df.copy()
        df_out["pseudo_label_idx"] = pseudo_label
        df_out["pseudo_label_str"] = [idx_to_style[int(i)] for i in pseudo_label]

        # class별 확률도 붙이고 싶으면:
        for style_name, c_idx in DBA_STYLE_TO_LABEL.items():
            df_out[f"pseudo_prob_{style_name}"] = prob[:, c_idx]

        # 저장 경로 (원본 구조와 동일하게 out_root 아래에 복사)
        rel_path = os.path.relpath(csv_path, args.dba_root)
        out_dir  = os.path.join(args.out_root, os.path.dirname(rel_path))
        os.makedirs(out_dir, exist_ok=True)

        out_csv_name = os.path.basename(csv_path).replace(
            "parsed_50hz", "parsed_50hz_with_pseudo"
        )
        out_csv = os.path.join(out_dir, out_csv_name)

        df_out.to_csv(out_csv, index=False)
        print(f"[Pseudo] saved: {out_csv}")


if __name__ == "__main__":
    main()
