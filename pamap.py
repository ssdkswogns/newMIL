# prepare_wearable_dat.py
# -*- coding: utf-8 -*-
import argparse, os, re, glob
import numpy as np
import pandas as pd

# ----- 고정 스키마: PAMAP2 .dat = 54열 (subject_id 없음) -----
COLS = [
    "timestamp","activityID","heartrate","handTemperature",
    "handAcc16_1","handAcc16_2","handAcc16_3",
    "handAcc6_1","handAcc6_2","handAcc6_3",
    "handGyro1","handGyro2","handGyro3",
    "handMagne1","handMagne2","handMagne3",
    "handOrientation1","handOrientation2","handOrientation3","handOrientation4",
    "chestTemperature",
    "chestAcc16_1","chestAcc16_2","chestAcc16_3",
    "chestAcc6_1","chestAcc6_2","chestAcc6_3",
    "chestGyro1","chestGyro2","chestGyro3",
    "chestMagne1","chestMagne2","chestMagne3",
    "chestOrientation1","chestOrientation2","chestOrientation3","chestOrientation4",
    "ankleTemperature",
    "ankleAcc16_1","ankleAcc16_2","ankleAcc16_3",
    "ankleAcc6_1","ankleAcc6_2","ankleAcc6_3",
    "ankleGyro1","ankleGyro2","ankleGyro3",
    "ankleMagne1","ankleMagne2","ankleMagne3",
    "ankleOrientation1","ankleOrientation2","ankleOrientation3","ankleOrientation4",
]
FEATURE_COLS = COLS[2:54]   # 2~53 → 52개
TARGET_COL   = "activityID"
TIME_COL     = "timestamp"
PAMAP2_LABELS = np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,16,17,18,19,20,24], dtype=np.int32)

def read_dat_return_sid(path: str):
    """헤더 없는 .dat(공백 구분) → (sid, df) 반환. df에는 subject_id 컬럼을 넣지 않음."""
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None, names=COLS)
    m = re.findall(r'(\d+)', os.path.basename(path))
    sid = int(m[-1]) if m else 0
    return sid, df

def fill_nan_one(df: pd.DataFrame) -> pd.DataFrame:
    """파일 하나(df)에 대해 timestamp 정렬 + ffill → bfill → 0"""
    g = df.sort_values(TIME_COL).copy()
    g[FEATURE_COLS] = g[FEATURE_COLS].ffill().bfill()
    g[FEATURE_COLS] = g[FEATURE_COLS].fillna(0.0)
    return g

def build_windows_from_files(sid_df_list, seq_len, step, label_values_sorted):
    label_to_idx = {int(v): i for i, v in enumerate(label_values_sorted)}
    X_list, Y_seq_list, sid_list, Yts_int_list = [], [], [], []

    for sid, df in sid_df_list:
        df = fill_nan_one(df)
        if len(df) < seq_len:
            continue
        feats_np  = df[FEATURE_COLS].to_numpy(dtype=np.float32)  # (T, 52)
        labels_np = df[TARGET_COL].to_numpy(dtype=int)            # (T,)

        for s in range(0, len(df) - seq_len + 1, step):
            e = s + seq_len
            window = feats_np[s:e]                 # (L, 52)
            y_ts_int = labels_np[s:e].astype(int)  # (L,)

            # 시퀀스 멀티-핫 라벨 (윈도우 내 고유 라벨 집합)
            uniq = np.unique(y_ts_int)
            idxs = [label_to_idx[u] for u in uniq if u in label_to_idx]
            if not idxs:
                continue
            y_seq = np.zeros(len(label_values_sorted), dtype=np.float32)
            y_seq[idxs] = 1.0

            X_list.append(window)
            Y_seq_list.append(y_seq)
            Yts_int_list.append(y_ts_int)
            sid_list.append(sid)

    if not X_list:
        return (np.empty((0, seq_len, len(FEATURE_COLS)), np.float32),
                np.empty((0, len(label_values_sorted)), np.float32),
                [], np.empty((0, seq_len), np.int32))

    X = np.stack(X_list, 0)
    Y_seq = np.stack(Y_seq_list, 0)
    Y_ts_int = np.stack(Yts_int_list, 0)  # (N, L)
    return X, Y_seq, sid_list, Y_ts_int

def split_by_subject(subject_ids, val_ratio, test_ratio, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(subject_ids)))
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    test_sub = set(uniq[:n_test])
    val_sub  = set(uniq[n_test:n_test+n_val])
    train_sub= set(uniq[n_test+n_val:])
    return train_sub, val_sub, test_sub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="subject별 .dat 파일들이 있는 폴더")
    ap.add_argument("--glob", type=str, default="*.dat", help="파일 글롭 패턴")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--assume_has_header", action="store_true", help=".dat에 헤더가 있다면 지정")
    ap.add_argument("--subject_from_name", action="store_true", help="파일명에서 subject_id 추출해서 사용")
    ap.add_argument("--save_path", type=str, required=True, help="저장할 npz 경로")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_root, args.glob)))
    assert paths, "no .dat files"

    # (sid, df) 목록 생성 (df에는 subject_id 컬럼 넣지 않음)
    sid_df_list = [read_dat_return_sid(p) for p in paths]

    label_values = PAMAP2_LABELS

    # 윈도우 생성
    X, Y, sid_list, Y_ts_int = build_windows_from_files(
        sid_df_list, seq_len=args.seq_len, step=args.step,
        label_values_sorted=label_values
    )
    print(f"Windows built: X={X.shape}, Y={Y.shape}, unique_subjects={len(set(sid_list))}")

    # subject-wise split (df에 subject_id가 없으므로 sid_list로 분할)
    train_sub, val_sub, test_sub = split_by_subject(sid_list, args.val_ratio, args.test_ratio, args.seed)
    sid_arr = np.array(sid_list)
    idx_tr = np.where(np.isin(sid_arr, list(train_sub)))[0]
    idx_va = np.where(np.isin(sid_arr, list(val_sub)))[0]
    idx_te = np.where(np.isin(sid_arr, list(test_sub)))[0]

    def take(idx):
        return X[idx], Y[idx], Y_ts_int[idx]

    X_tr, Y_tr, Yts_tr = take(idx_tr)
    X_va, Y_va, Yts_va = take(idx_va)
    X_te, Y_te, Yts_te = take(idx_te)
    print(f"Split -> train: {X_tr.shape}, val: {X_va.shape}, test: {X_te.shape}")

    # 저장 (subject_id는 저장하지 않음)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.savez_compressed(
        args.save_path,
        X_train=X_tr, y_train=Y_tr, y_ts_train=Yts_tr,   # ← ★ y_ts_* 저장
        X_val=X_va,   y_val=Y_va,  y_ts_val=Yts_va,
        X_test=X_te,  y_test=Y_te, y_ts_test=Yts_te,
        label_values=label_values.astype(np.int32),
        seq_len=np.int32(args.seq_len),
        feat_dim=np.int32(len(FEATURE_COLS)),
        feature_names=np.array(FEATURE_COLS),
    )
    print(f"Saved: {args.save_path}")

if __name__ == "__main__":
    main()
