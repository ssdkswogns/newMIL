# visualize_dba_pseudo.py
# -*- coding: utf-8 -*-

import os
import argparse
import glob

import numpy as np
import pandas as pd

from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt

# 있으면 쓰고, 없으면 fallback
try:
    from dba_dataloader import DBA_FEATURE_COLS, DBA_STYLE_TO_LABEL
    DEFAULT_FEATURE_COLS = DBA_FEATURE_COLS
    DEFAULT_LABEL_NAMES = list(DBA_STYLE_TO_LABEL.keys())
except Exception:
    DEFAULT_FEATURE_COLS = [
        "imu_acc_long", "imu_acc_lat", "imu_yaw_rate","imu_roll_rate",
        "odom_vx",
        "odom_wz",
        "radar_obj_relspd",
        "radar_obj_dst",
    ]
    DEFAULT_LABEL_NAMES = ["aggressive", "conservative", "normal"]

# --- 추가: pseudo prob 컬럼 이름 & low-confidence threshold ---
PSEUDO_PROB_COLS = [
    "pseudo_prob_aggressive",
    "pseudo_prob_conservative",
    "pseudo_prob_normal",
]
PSEUDO_CLASSES = ["aggressive", "conservative", "normal"]
LOW_CONF_LABEL = "low_conf"
LOW_CONF_THRESH = 0.5   # 세 확률 모두 이 값 미만이면 low_conf로 처리


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_root",
        type=str,
        required=True,
        help="pseudo label이 포함된 csv들이 있는 root 디렉토리 (예: /mnt/storage2/dba_parser_pseudo)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="이미지를 저장할 root 디렉토리",
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default=None,
        help="시간 컬럼 이름 (없으면 index를 시간축으로 사용)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="pseudo_label_str",
        help="(기존) string pseudo label 컬럼 이름 (지금은 무시하고, prob 기반으로 다시 계산)",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=None,
        help="플롯할 feature 컬럼 이름 리스트 (지정 안 하면 DBA_FEATURE_COLS 사용)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="저장할 이미지 DPI",
    )
    return parser.parse_args()


def _build_label_color_map(unique_labels):
    """
    label string -> color 매핑.
    aggressive / conservative / normal 기준으로 직관적인 색.
    low_conf(저신뢰)는 검정색으로 표시.
    나머지는 cycle에서 자동 할당.
    """
    base_colors = {
        "aggressive": "red",
        "conservative": "blue",
        "normal": "green",
        LOW_CONF_LABEL: "black",
    }

    # matplotlib default color cycle
    cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cycle_idx = 0

    label2color = {}
    for lab in unique_labels:
        if lab in base_colors:
            label2color[lab] = base_colors[lab]
        else:
            label2color[lab] = cycler[cycle_idx % len(cycler)]
            cycle_idx += 1

    return label2color


def plot_sequence_with_pseudo(
    df,
    out_path,
    time_col=None,
    feature_cols=None,
    label_col="pseudo_label_str",
    dpi=150,
):
    # 0) pseudo prob 컬럼 체크
    for col in PSEUDO_PROB_COLS:
        if col not in df.columns:
            raise ValueError(f"필요한 컬럼 {col} 이(가) CSV에 없습니다: {out_path}")

    # 1) 사용할 feature 선택
    if feature_cols is None:
        feature_cols = []
        col_map = {}   # 표시이름 -> 실제 df 컬럼

        for c in DEFAULT_FEATURE_COLS:
            if c in df.columns:
                feature_cols.append(c)
                col_map[c] = c

            elif c == "radar_obj_relspd" and "scc_obj_relspd" in df.columns:
                feature_cols.append("radar_obj_relspd")
                col_map["radar_obj_relspd"] = "scc_obj_relspd"

            elif c == "radar_obj_dst" and "scc_obj_dst" in df.columns:
                feature_cols.append("radar_obj_dst")
                col_map["radar_obj_dst"] = "scc_obj_dst"

    if len(feature_cols) == 0:
        raise ValueError("플롯할 feature 컬럼이 없습니다. --features 또는 DEFAULT_FEATURE_COLS를 확인하세요.")

    n_feat = len(feature_cols)

    # 2) 시간축
    if time_col is not None and time_col in df.columns:
        t = df[time_col].values
    else:
        t = np.arange(len(df))

    # 3) feature 값
    y_dict = {}
    for feat in feature_cols:
        real_col = col_map.get(feat, feat)
        y_dict[feat] = df[real_col].values

    # 4) pseudo prob → pseudo label_str 재계산
    #    probs: [T, 3]
    prob_aggr = df[PSEUDO_PROB_COLS[0]].to_numpy(dtype=float)
    prob_cons = df[PSEUDO_PROB_COLS[1]].to_numpy(dtype=float)
    prob_norm = df[PSEUDO_PROB_COLS[2]].to_numpy(dtype=float)

    probs = np.stack([prob_aggr, prob_cons, prob_norm], axis=-1)   # [T, 3]
    max_prob = probs.max(axis=1)                                   # [T]
    argmax_idx = probs.argmax(axis=1)                              # [T]

    labels_str = []
    for i in range(len(probs)):
        if max_prob[i] < LOW_CONF_THRESH:
            labels_str.append(LOW_CONF_LABEL)
        else:
            labels_str.append(PSEUDO_CLASSES[argmax_idx[i]])
    labels_str = np.array(labels_str, dtype=str)

    # 5) downsampling (지금은 전체 사용, 단 x축 tick만 줄임)
    T = len(t)
    idx = np.arange(T)

    t_ds = t[idx]
    y_ds = {k: v[idx] for k, v in y_dict.items()}
    labels_ds = labels_str[idx]
    prob_aggr_ds = prob_aggr[idx]
    prob_cons_ds = prob_cons[idx]
    prob_norm_ds = prob_norm[idx]

    # 6) label → color 매핑
    unique_labels = sorted(list(set(labels_ds)))
    label2color = _build_label_color_map(unique_labels)
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    label_ids = np.array([label2id[lab] for lab in labels_ds], dtype=np.int32)[None, :]  # (1, T_ds)

    from matplotlib.colors import ListedColormap
    cmap_colors = [label2color[lab] for lab in unique_labels]
    cmap = ListedColormap(cmap_colors)

    # 7) figure / axes 구성
    #   - 위에 n_feat개의 시계열 subplot
    #   - 그 아래 1개: pseudo probs (3개)
    #   - 맨 아래 1개: label band
    height_ratios = [2.0] * n_feat + [1.2, 0.6]
    fig, axes = plt.subplots(
        n_feat + 2,
        1,
        sharex=True,
        figsize=(16, 2 * n_feat + 3),
        gridspec_kw={"height_ratios": height_ratios},
        dpi=dpi,
    )

    # feature 축들: 0 ~ n_feat-1
    # prob 축: n_feat
    # label band: n_feat+1
    axes_feat = axes[:n_feat]
    ax_prob = axes[n_feat]
    ax_lbl = axes[n_feat + 1]

    # 배경을 흰색으로
    for ax in axes:
        ax.set_facecolor("white")

    # 8) feature별 subplot에 시계열 그리기
    for i, feat in enumerate(feature_cols):
        ax = axes_feat[i]
        ax.plot(t_ds, y_ds[feat], linewidth=0.8)
        ax.set_ylabel(feat, fontsize=9)
        ax.grid(True, alpha=0.3)

        # 맨 위 축만 title
        if i == 0:
            ax.set_title(os.path.basename(out_path), fontsize=10)

        # x tick label은 아래 축에서만 표시
        if i < n_feat:
            ax.tick_params(labelbottom=False)

    # 9) pseudo prob subplot
    # 각 클래스별 색은 label band 색과 맞춤
    if "aggressive" in label2color:
        c_aggr = label2color["aggressive"]
    else:
        c_aggr = "red"
    if "conservative" in label2color:
        c_cons = label2color["conservative"]
    else:
        c_cons = "blue"
    if "normal" in label2color:
        c_norm = label2color["normal"]
    else:
        c_norm = "green"

    ax_prob.plot(t_ds, prob_aggr_ds, label="aggressive", linewidth=0.8, color=c_aggr)
    ax_prob.plot(t_ds, prob_cons_ds, label="conservative", linewidth=0.8, color=c_cons)
    ax_prob.plot(t_ds, prob_norm_ds, label="normal", linewidth=0.8, color=c_norm)
    ax_prob.set_ylabel("prob", fontsize=9)
    ax_prob.set_ylim(0.0, 1.0)
    ax_prob.grid(True, alpha=0.3)
    ax_prob.legend(loc="upper right", fontsize=8)
    ax_prob.tick_params(labelbottom=False)

    # 10) label band 그리기
    x_min, x_max = t_ds[0], t_ds[-1]
    ax_lbl.imshow(
        label_ids,
        aspect="auto",
        interpolation="nearest",
        extent=[x_min, x_max, 0, 1],
        cmap=cmap,
    )
    ax_lbl.set_yticks([])
    ax_lbl.set_ylim(0, 1)

    if time_col is not None and time_col in df.columns:
        ax_lbl.set_xlabel(time_col)
    else:
        ax_lbl.set_xlabel("time index")

    # label legend
    handles = []
    for lab in unique_labels:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=label2color[lab],
                markersize=8,
                label=lab,
            )
        )
    ax_lbl.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=len(unique_labels),
        fontsize=8,
        frameon=False,
    )
    # x-axis tick 듬성듬성
    ax_lbl.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    in_root = args.in_root
    out_root = args.out_root
    time_col = args.time_col
    label_col = args.label_col
    feature_cols = args.features
    dpi = args.dpi

    # in_root 아래 모든 csv 탐색
    pattern = os.path.join(in_root, "**", "*.csv")
    csv_paths = sorted(glob.glob(pattern, recursive=True))

    if len(csv_paths) == 0:
        print(f"[WARN] No csv found under: {in_root}")
        return

    print(f"[INFO] Found {len(csv_paths)} csv files under {in_root}")

    for csv_path in csv_paths:
        print(f"[VIS] {csv_path}")
        try:
            df = pd.read_csv(csv_path)

            # 출력 경로: in_root 기준 상대경로 유지
            rel_path = os.path.relpath(csv_path, in_root)
            rel_dir = os.path.dirname(rel_path)
            base = os.path.splitext(os.path.basename(csv_path))[0]

            out_dir = os.path.join(out_root, rel_dir)
            out_path = os.path.join(out_dir, base + ".png")

            plot_sequence_with_pseudo(
                df,
                out_path=out_path,
                time_col=time_col,
                feature_cols=feature_cols,
                label_col=label_col,
                dpi=dpi,
            )
        except Exception as e:
            print(f"[ERROR] Failed to visualize {csv_path}: {e}")


if __name__ == "__main__":
    main()
