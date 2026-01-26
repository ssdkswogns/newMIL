# -*- coding: utf-8 -*-

from html import parser
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
#from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,f1_score,accuracy_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

import torch.distributed as dist

# from models.dropout import LinearScheduler
from aeon.datasets import load_classification
from syntheticdataset import *
from utils import *
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from mydataload import loadorean
from dba_dataloader import build_dba_for_timemil, build_dba_windows_for_mixed
import random

import wandb

# from timm.optim.adamp import AdamP
from lookhead import Lookahead
import warnings

from models.timemil_old import newTimeMIL, AmbiguousMIL
from models.timemil import TimeMIL
from models.expmil import AmbiguousMILwithCL
from compute_aopcr import compute_classwise_aopcr

from os.path import join

# Suppress all warnings
warnings.filterwarnings("ignore")

import math

from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw, dtw_path, gak
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def get_MDTW_blockwise_memmap(
    MTS_tr,
    memmap_path: str,
    block_size: int = 256,
    resume: bool = True,
    checkpoint_path: str = None,
    verbose: bool = True,
):
    """
    MTS_tr: list of np arrays, each [T, D] float32
    dist_mat를 (N,N) memmap으로 디스크에 생성/재사용하며
    블록 단위로 DTW를 계산해 채웁니다.

    - resume=True: 이미 계산된 블록은 스킵 (checkpoint 기반)
    - checkpoint: 계산된 row-block index를 저장 (간단/안정)
    """
    N = len(MTS_tr)
    os.makedirs(os.path.dirname(os.path.abspath(memmap_path)), exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = memmap_path + ".ckpt.npy"

    # memmap 생성/로드
    if os.path.exists(memmap_path):
        dist_mm = np.memmap(memmap_path, dtype=np.float32, mode="r+", shape=(N, N))
    else:
        dist_mm = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(N, N))
        # 초기값: diag=0, 나머지=-1 (미계산 마킹)
        dist_mm[:] = -1.0
        np.fill_diagonal(dist_mm, 0.0)
        dist_mm.flush()

    # checkpoint 로드
    done_blocks = set()
    if resume and os.path.exists(checkpoint_path):
        done = np.load(checkpoint_path, allow_pickle=True).tolist()
        done_blocks = set(done) if isinstance(done, list) else set()

    n_blocks = math.ceil(N / block_size)

    def save_ckpt():
        np.save(checkpoint_path, sorted(list(done_blocks)))

    t0 = time.time()
    for bi in range(n_blocks):
        if resume and (bi in done_blocks):
            continue

        i0 = bi * block_size
        i1 = min(N, (bi + 1) * block_size)

        if verbose:
            print(f"[DTW] block {bi+1}/{n_blocks} rows [{i0}:{i1})  elapsed={time.time()-t0:.1f}s", flush=True)

        # j-block은 0..bi까지만 (하삼각)
        for bj in range(bi + 1):
            j0 = bj * block_size
            j1 = min(N, (bj + 1) * block_size)

            # 블록 내부 채우기
            for i in range(i0, i1):
                # 대칭/중복 방지: j는 <= i
                j_max = min(j1, i + 1) if bj == bi else j1
                for j in range(j0, j_max):
                    if i == j:
                        continue
                    # resume 시 이미 값이 있으면 스킵
                    if resume and dist_mm[i, j] >= 0:
                        continue
                    d = dtw(MTS_tr[i], MTS_tr[j])
                    dist_mm[i, j] = d
                    dist_mm[j, i] = d  # 대칭 채우기

        dist_mm.flush()
        done_blocks.add(bi)
        save_ckpt()

    return dist_mm  # memmap handle

def global_minmax_offdiag(dist_mm, block=1024):
    N = dist_mm.shape[0]
    gmin = np.inf
    gmax = -np.inf
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        chunk = np.array(dist_mm[i0:i1, :], copy=False)  # view 가능
        # 대각 제외: i0..i1 구간의 diag 위치만 마스킹
        for ii in range(i0, i1):
            chunk[ii - i0, ii] = np.nan
        cmin = np.nanmin(chunk)
        cmax = np.nanmax(chunk)
        gmin = min(gmin, cmin)
        gmax = max(gmax, cmax)
    return gmin, gmax


def _atomic_save_npz(out_path: str, **kwargs):
    _ensure_parent_dir(out_path)
    tmp_path = out_path + ".tmp"
    np.savez_compressed(tmp_path, **kwargs)
    os.replace(tmp_path, out_path)  # atomic (same filesystem)

def _wait_for_file(done_path: str, timeout_sec: int = 24*3600, poll_sec: float = 2.0):
    t0 = time.time()
    while True:
        if os.path.exists(done_path):
            return
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"[SoftCLT] Timed out waiting for: {done_path}")
        time.sleep(poll_sec)

def prepare_softclt_sim_matrix(
    sim_mat_path: str,
    trainset,
    args,
    is_main: bool,
    world_size: int,
):
    """
    sim_mat_path가 없으면 trainset으로 soft similarity matrix를 생성.
    DDP 환경에서 rank 0만 생성.
    다른 rank들은 dist.barrier() 대신 파일 생성 완료(done flag)까지 폴링.
    """
    # done_flag = sim_mat_path + ".done"

    if os.path.exists(sim_mat_path):
        if is_main:
            print(f"[SoftCLT] Found existing sim matrix: {sim_mat_path}")
    else:
        if is_main:
            print(f"[SoftCLT] sim matrix not found. Building: {sim_mat_path}")

            # ---- 여기서 생성 ----
            soft_mat = build_softclt_soft_matrix_from_dataset(
                trainset,
                type_=args.dist_metric,
                min_=args.dist_min,
                max_=args.dist_max,
                multivariate=True,
                densify_tau=args.densify_tau,
                densify_alpha=args.densify_alpha,
                out_path=None,  # 아래에서 atomic 저장으로 처리
            )

            # atomic save + done flag
            _atomic_save_npz(
                sim_mat_path,
                soft_matrix=soft_mat,
                metric=args.dist_metric,
                multivariate=True,
                min_=args.dist_min,
                max_=args.dist_max,
                densify_tau=args.densify_tau,
                densify_alpha=args.densify_alpha
            )
            with open(done_flag, "w") as f:
                f.write("ok\n")

        else:
            # rank0가 저장을 끝낼 때까지 폴링
            _wait_for_file(done_flag, timeout_sec=24*3600, poll_sec=2.0)

    # ---- 로드 ----
    npz = np.load(sim_mat_path)
    if "soft_matrix" in npz:
        mat = npz["soft_matrix"]
    elif "sim_mat" in npz:
        mat = npz["sim_mat"]
    else:
        raise KeyError(
            f"[SoftCLT] {sim_mat_path} keys={list(npz.keys())} "
            "중 soft_matrix/sim_mat 없음"
        )

    return torch.from_numpy(mat).float().cpu()

def _ensure_parent_dir(filepath: str):
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent and (not os.path.exists(parent)):
        os.makedirs(parent, exist_ok=True)

def _to_tslearn_univar(x):
    """
    SoftCLT 원본은 UTS: shape [T] 를 reshape(-1,1)로 dtw에 넣습니다.
    여기서는 x가 torch/np + [T, D] 혹은 [D, T] 혹은 [T]일 수 있으므로
    "univariate 거리"를 만들기 위해 D>1이면 (권장) feature 평균으로 1D로 줄입니다.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    if x.ndim == 1:
        uts = x.astype(np.float32, copy=False)
    elif x.ndim == 2:
        # [T,D] or [D,T]
        # time axis를 크게 보고, 그 방향을 T로 가정
        if x.shape[0] < x.shape[1] and x.shape[1] > 32:
            x = x.T  # [T,D]로 맞춤
        # D>1이면 univariate로 평균(SoftCLT UTS 버전 호환 목적)
        uts = x.mean(axis=1).astype(np.float32, copy=False)  # [T]
    else:
        raise ValueError(f"Unsupported feats shape: {x.shape}")

    return uts.reshape(-1, 1)  # tslearn dtw 입력 형태: [T,1]

def _to_tslearn_mts(x):
    """
    입력 feats를 tslearn multivariate DTW 입력 형태 [T, D] (float32)로 변환.
    - x: torch/np, shape could be [T,D] or [D,T] or [T]
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    if x.ndim == 1:
        # 단변량도 [T,1]로 확장해 multivariate 루틴에서 일관 처리 가능
        mts = x.reshape(-1, 1).astype(np.float32, copy=False)
        return mts

    if x.ndim != 2:
        raise ValueError(f"Unsupported feats shape for MTS: {x.shape}")

    # [D,T] vs [T,D] 추정: 보통 T가 더 큼
    if x.shape[0] < x.shape[1] and x.shape[1] > 32:
        x = x.T  # [T,D]로 맞춤

    return x.astype(np.float32, copy=False)  # [T,D]

def get_DTW(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1):
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                d = dtw(UTS_tr[i], UTS_tr[j])
                dist_mat[i, j] = d
                dist_mat[j, i] = d
    return dist_mat

def get_MDTW(MTS_tr):
    """
    MTS_tr: list of np arrays, each [T, D]
    """
    N = len(MTS_tr)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1):
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                d = dtw(MTS_tr[i], MTS_tr[j])  # tslearn multivariate DTW
                dist_mat[i, j] = d
                dist_mat[j, i] = d
    return dist_mat

def get_TAM(UTS_tr):
    def find(condition):
        res, = np.nonzero(np.ravel(condition))
        return res

    def tam(path):
        delay = len(find(np.diff(path[0]) == 0))
        advance = len(find(np.diff(path[1]) == 0))
        incumbent = find((np.diff(path[0]) == 1) * (np.diff(path[1]) == 1))
        phase = len(incumbent)

        len_estimation = path[1][-1]
        len_ref = path[0][-1]

        p_advance = advance * 1.0 / len_ref
        p_delay = delay * 1.0 / len_estimation
        p_phase = phase * 1.0 / np.min([len_ref, len_estimation])
        return p_advance + p_delay + (1.0 - p_phase)

    N = len(UTS_tr)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1):
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                path = dtw_path(UTS_tr[i], UTS_tr[j])[0]
                p = [np.array([a for a, _ in path]), np.array([b for _, b in path])]
                d = tam(p)
                dist_mat[i, j] = d
                dist_mat[j, i] = d
    return dist_mat

def get_GAK(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1):
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                d = gak(UTS_tr[i], UTS_tr[j])
                dist_mat[i, j] = d
                dist_mat[j, i] = d
    return dist_mat

def get_COS(MTS_tr_flat):
    # SoftCLT 코드: cos_sim_matrix = -cosine_similarity(MTS_tr)
    # 여기서는 "distance"가 필요하면 negative cosine similarity를 dist로 사용
    return (-cosine_similarity(MTS_tr_flat)).astype(np.float32)

def get_EUC(MTS_tr_flat):
    return euclidean_distances(MTS_tr_flat).astype(np.float32)

def save_sim_mat(X_tr, min_=0.0, max_=1.0, multivariate=False, type_='DTW'):
    """
    SoftCLT 원본 save_sim_mat 동일 로직 + multivariate DTW 지원.
    """
    if multivariate:
        # SoftCLT 원본 코드에서는 multivariate일 때 DTW만 사용(원본 assert 의도)
        if type_ != 'DTW':
            raise ValueError("For multivariate=True, only type_='DTW' is supported (SoftCLT original behavior).")
        dist_mat = get_MDTW(X_tr)  # X_tr is list of [T,D]
    else:
        if type_ == 'DTW':
            dist_mat = get_DTW(X_tr)   # X_tr is list of [T,1]
        elif type_ == 'TAM':
            dist_mat = get_TAM(X_tr)
        elif type_ == 'GAK':
            dist_mat = get_GAK(X_tr)
        elif type_ == 'COS':
            dist_mat = get_COS(X_tr)
        elif type_ == 'EUC':
            dist_mat = get_EUC(X_tr)
        else:
            raise ValueError(f"Unknown type_: {type_}")

    N = dist_mat.shape[0]

    # (1) distance matrix: diag를 off-diag 최소로 채움 (원본과 동일)
    diag_indices = np.diag_indices(N)
    mask = np.ones(dist_mat.shape, dtype=bool)
    mask[diag_indices] = False
    temp = dist_mat[mask].reshape(N, N - 1)
    dist_mat[diag_indices] = temp.min()

    # (2) normalized distance matrix
    scaler = MinMaxScaler(feature_range=(min_, max_))
    dist_mat = scaler.fit_transform(dist_mat)

    # (3) normalized similarity matrix
    sim_mat = (1.0 - dist_mat).astype(np.float32)
    return sim_mat

def densify(sim_mat, tau, alpha):
    # 원본과 동일
    return ((2 * alpha) / (1 + np.exp(-tau * sim_mat))) + (1 - alpha) * np.eye(sim_mat.shape[0], dtype=np.float32)

def topK_one_else_zero(matrix, k):
    new_matrix = matrix.copy()
    top_k_indices = np.argpartition(new_matrix, -(k + 1), axis=1)[:, -(k + 1):]
    np.fill_diagonal(new_matrix, 0)
    mask = np.zeros_like(new_matrix)
    mask[np.repeat(np.arange(new_matrix.shape[0]), (k + 1)), top_k_indices.flatten()] = 1
    return mask

def convert_hard_matrix(soft_matrix, pos_ratio):
    N = soft_matrix.shape[0]
    num_pos = int((N - 1) * pos_ratio)
    return topK_one_else_zero(soft_matrix, num_pos)

# ---------------------------
# CPU 폭주 방지 (선택 권장)
# ---------------------------
def limit_cpu_threads(n: int = 1):
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))

def _sanitize_np(x: np.ndarray) -> np.ndarray:
    # dtw에 NaN/Inf가 들어가면 터질 수 있으므로 방어
    x = np.asarray(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.astype(np.float32, copy=False)
    return np.ascontiguousarray(x)

def _to_tslearn_mts_safe(feats):
    mts = _to_tslearn_mts(feats)  # 기존 사용
    return _sanitize_np(mts)

def _to_tslearn_univar_safe(feats):
    uts = _to_tslearn_univar(feats)  # 기존 사용
    return _sanitize_np(uts)

def _ensure_parent_dir(filepath: str):
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent and (not os.path.exists(parent)):
        os.makedirs(parent, exist_ok=True)

def _make_memmap(path: str, shape, mode="w+"):
    _ensure_parent_dir(path)
    return np.memmap(path, dtype=np.float32, mode=mode, shape=shape)

def get_MDTW_blockwise_memmap(
    X_list,                       # list of [T,D] float32
    dist_memmap_path: str,
    block_size: int = 128,
    resume: bool = True,
    ckpt_path: str = None,
    verbose: bool = True,
):
    """
    DTW 거리행렬 NxN을 memmap(디스크)로 만들고 블록 단위로 채움.
    - 계산법(DTW) 동일
    - RAM 사용량 최소화
    - 중간에 죽어도 resume 가능
    """
    N = len(X_list)
    if ckpt_path is None:
        ckpt_path = dist_memmap_path + ".ckpt.npy"

    # dist memmap 생성/로드
    if os.path.exists(dist_memmap_path):
        dist_mm = _make_memmap(dist_memmap_path, (N, N), mode="r+")
    else:
        dist_mm = _make_memmap(dist_memmap_path, (N, N), mode="w+")
        dist_mm[:] = -1.0
        np.fill_diagonal(dist_mm, 0.0)
        dist_mm.flush()

    done_blocks = set()
    if resume and os.path.exists(ckpt_path):
        done = np.load(ckpt_path, allow_pickle=True).tolist()
        if isinstance(done, list):
            done_blocks = set(done)

    n_blocks = math.ceil(N / block_size)
    t0 = time.time()

    def save_ckpt():
        np.save(ckpt_path, sorted(list(done_blocks)))

    for bi in range(n_blocks):
        if resume and (bi in done_blocks):
            continue

        i0 = bi * block_size
        i1 = min(N, (bi + 1) * block_size)

        if verbose:
            print(f"[DTW] block {bi+1}/{n_blocks} rows [{i0}:{i1})  elapsed={time.time()-t0:.1f}s", flush=True)

        # 하삼각만 계산: bj <= bi
        for bj in range(bi + 1):
            j0 = bj * block_size
            j1 = min(N, (bj + 1) * block_size)

            for i in range(i0, i1):
                j_max = min(j1, i + 1) if bj == bi else j1
                for j in range(j0, j_max):
                    if i == j:
                        continue
                    # 이미 계산했으면 스킵
                    if resume and dist_mm[i, j] >= 0:
                        continue
                    try:
                        d = dtw(X_list[i], X_list[j])
                    except Exception:
                        # 한 쌍 때문에 전체가 죽지 않도록 매우 큰 거리로 대체
                        d = 1e6
                    dist_mm[i, j] = d
                    dist_mm[j, i] = d  # 대칭

        dist_mm.flush()
        done_blocks.add(bi)
        save_ckpt()

    return dist_mm

def _rowwise_offdiag_min(dist_mm, block_size: int = 1024) -> np.ndarray:
    """
    각 row i에 대해 min_{j!=i} dist[i,j]를 구함. (diag 제외)
    dist_mm은 memmap이어도 동작.
    """
    N = dist_mm.shape[0]
    row_min = np.full((N,), np.inf, dtype=np.float32)
    for i0 in range(0, N, block_size):
        i1 = min(N, i0 + block_size)
        chunk = np.array(dist_mm[i0:i1, :], copy=False)  # view 가능
        # diag 제외: chunk 내부에서 해당 위치만 inf 처리
        for ii in range(i0, i1):
            chunk[ii - i0, ii] = np.inf
        row_min[i0:i1] = np.min(chunk, axis=1).astype(np.float32)
    return row_min

def _global_minmax(dist_mm, block_size: int = 1024):
    """
    전체 행렬의 min/max(대각 포함)를 구하되,
    여기서는 diag를 이미 row_min으로 바꾼 뒤 사용할 것을 권장.
    """
    N = dist_mm.shape[0]
    gmin = np.inf
    gmax = -np.inf
    for i0 in range(0, N, block_size):
        i1 = min(N, i0 + block_size)
        chunk = np.array(dist_mm[i0:i1, :], copy=False)
        cmin = float(np.min(chunk))
        cmax = float(np.max(chunk))
        gmin = min(gmin, cmin)
        gmax = max(gmax, cmax)
    return gmin, gmax

def _write_soft_matrix_from_dist_memmap(
    dist_mm,
    out_soft_memmap_path: str,
    min_: float,
    max_: float,
    block_size: int = 1024,
    verbose: bool = True,
):
    """
    SoftCLT 원본 로직 유지:
      1) diag = row offdiag min
      2) MinMaxScaler(feature_range=(min_, max_)) == affine scaling
      3) sim = 1 - dist_norm
    단, RAM 폭발 방지를 위해 블록 처리 + memmap 저장
    """
    N = dist_mm.shape[0]

    # (1) diag 보정: d_ii <- min_{j!=i} d_ij
    if verbose:
        print("[SoftCLT] computing row-wise off-diagonal minima for diag fill ...", flush=True)
    row_min = _rowwise_offdiag_min(dist_mm, block_size=block_size)

    if verbose:
        print("[SoftCLT] filling diagonal with row minima ...", flush=True)
    for i0 in range(0, N, block_size):
        i1 = min(N, i0 + block_size)
        # diag만 블록으로 채움
        for ii in range(i0, i1):
            dist_mm[ii, ii] = row_min[ii]
    dist_mm.flush()

    # (2) global min/max (feature_range scaling을 위한)
    if verbose:
        print("[SoftCLT] computing global min/max ...", flush=True)
    dmin, dmax = _global_minmax(dist_mm, block_size=block_size)
    denom = (dmax - dmin) if (dmax > dmin) else 1.0

    # MinMaxScaler(feature_range=(min_, max_))와 동일:
    # scaled = (x - dmin)/denom * (max_-min_) + min_
    a = (max_ - min_) / denom
    b = min_ - dmin * a

    # (3) dist_norm -> sim (1 - dist_norm) 을 soft memmap에 작성
    soft_mm = _make_memmap(out_soft_memmap_path, (N, N), mode="w+")
    if verbose:
        print("[SoftCLT] writing normalized similarity matrix (streaming) ...", flush=True)

    for i0 in range(0, N, block_size):
        i1 = min(N, i0 + block_size)
        chunk = np.array(dist_mm[i0:i1, :], copy=False).astype(np.float32, copy=False)
        dist_norm = chunk * a + b
        sim = (1.0 - dist_norm).astype(np.float32)
        soft_mm[i0:i1, :] = sim
        soft_mm.flush()

    return soft_mm  # memmap handle

def build_softclt_soft_matrix_from_dataset(
    dataset,
    type_='DTW',
    min_=0.0,
    max_=1.0,
    multivariate=True,
    densify_tau=None,
    densify_alpha=None,
    out_path=None,
    # --- 추가 옵션(안정화) ---
    dtw_block_size: int = 128,
    norm_block_size: int = 1024,
    resume: bool = True,
    tmp_dir: str = "./softclt_cache",
    cpu_threads: int = 1,
    verbose: bool = True,
):
    """
    기존 계산법 유지:
      dataset -> X(list) -> (DTW NxN dist) -> diag fill -> MinMax -> sim=1-dist -> (optional) densify -> npz 저장

    변경점:
      - DTW 거리행렬을 memmap으로 블록 계산(디스크 저장)
      - 정규화/유사도 생성도 블록 스트리밍(memmap)
      - densify는 마지막에 필요 시 RAM으로 올려 처리(여전히 NxN이므로 매우 크면 부담)
    """
    if type_ != 'DTW':
        raise ValueError("현재 버전은 계산법 유지를 위해 DTW만 지원하도록 구성했습니다.")
    if not multivariate:
        raise ValueError("요청하신 '현재 계산법 유지' 기준에서 large dataset 문제는 multivariate DTW가 핵심이라 여기서는 multivariate=True만 다룹니다.")

    limit_cpu_threads(cpu_threads)

    N = len(dataset)
    if verbose:
        print(f"[SoftCLT] N={N}  multivariate={multivariate}  metric={type_}", flush=True)

    # 1) X 만들기 (기존과 동일)
    X = []
    for i in range(N):
        item = dataset[i]
        feats = item[0]
        if multivariate:
            X.append(_to_tslearn_mts_safe(feats))      # [T,D]
        else:
            X.append(_to_tslearn_univar_safe(feats))   # [T,1]

    # 2) dist memmap 경로
    _ensure_parent_dir(tmp_dir + "/dummy")
    dist_memmap_path = os.path.join(tmp_dir, f"dist_{type_}_N{N}.mmap")
    soft_memmap_path = os.path.join(tmp_dir, f"soft_{type_}_N{N}.mmap")

    # 3) DTW 거리행렬 계산(블록/재개 가능)
    dist_mm = get_MDTW_blockwise_memmap(
        X,
        dist_memmap_path=dist_memmap_path,
        block_size=dtw_block_size,
        resume=resume,
        verbose=verbose,
    )

    # 4) dist -> soft(similarity) memmap 생성(원본 로직 유지, 스트리밍)
    soft_mm = _write_soft_matrix_from_dist_memmap(
        dist_mm,
        out_soft_memmap_path=soft_memmap_path,
        min_=min_,
        max_=max_,
        block_size=norm_block_size,
        verbose=verbose,
    )

    # 5) densify (원본과 동일)  ※주의: 여기서 NxN을 RAM으로 올립니다.
    if densify_tau is not None and densify_alpha is not None:
        if verbose:
            print("[SoftCLT] densify enabled (will materialize NxN in RAM) ...", flush=True)
        soft_mat = np.array(soft_mm, copy=True)  # RAM
        soft_mat = densify(soft_mat, densify_tau, densify_alpha).astype(np.float32)
    else:
        # npz 저장을 위해서는 결국 RAM array가 필요합니다.
        # (np.savez_compressed는 memmap을 직접 효율적으로 압축 저장하지 못합니다.)
        soft_mat = np.array(soft_mm, copy=True)

    # 6) 저장 (기존과 동일)
    if out_path is not None:
        _ensure_parent_dir(out_path)
        np.savez_compressed(
            out_path,
            soft_matrix=soft_mat,
            metric=type_,
            multivariate=multivariate,
            min_=min_,
            max_=max_,
            densify_tau=densify_tau,
            densify_alpha=densify_alpha
        )
        print(f"[dist_mat] saved soft_matrix: {out_path}  shape={soft_mat.shape}")

    return soft_mat

# def build_softclt_soft_matrix_from_dataset(dataset, type_='DTW', min_=0.0, max_=1.0,
#                                            multivariate=True,
#                                            densify_tau=None, densify_alpha=None,
#                                            out_path=None):
#     """
#     dataset -> X_tr(list) -> save_sim_mat -> (optional) densify -> 저장
#     multivariate=True이면 feats를 [T,D]로 만들어 MDTW(dist) 계산
#     """
#     N = len(dataset)

#     X = []
#     for i in range(N):
#         item = dataset[i]
#         feats = item[0]
#         if multivariate:
#             X.append(_to_tslearn_mts(feats))       # [T,D]
#         else:
#             X.append(_to_tslearn_univar(feats))    # [T,1]

#     soft_mat = save_sim_mat(X, min_=min_, max_=max_, multivariate=multivariate, type_=type_)

#     if densify_tau is not None and densify_alpha is not None:
#         soft_mat = densify(soft_mat, densify_tau, densify_alpha).astype(np.float32)

#     if out_path is not None:
#         _ensure_parent_dir(out_path)
#         np.savez_compressed(
#             out_path,
#             soft_matrix=soft_mat,
#             metric=type_,
#             multivariate=multivariate,
#             min_=min_,
#             max_=max_,
#             densify_tau=densify_tau,
#             densify_alpha=densify_alpha
#         )
#         print(f"[dist_mat] saved soft_matrix: {out_path}  shape={soft_mat.shape}")

#     return soft_mat

# ----- DDP utils -----
def setup_distributed():
    """
    torchrun --nproc_per_node=N 으로 실행하면
    LOCAL_RANK, RANK, WORLD_SIZE 환경 변수가 자동으로 설정됨.
    없으면 single GPU로 동작.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        is_main = (rank == 0)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_main = True
    return rank, world_size, local_rank, is_main

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# ---------------------------------------------------------------------
#   Seed
# ---------------------------------------------------------------------
seed = 42

random.seed(seed)             # python random
np.random.seed(seed)          # numpy random
torch.manual_seed(seed)       # CPU
torch.cuda.manual_seed(seed)  # GPU 단일
torch.cuda.manual_seed_all(seed)  # multi-GPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class WithIndexDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        out = self.base_ds[idx]
        idx_t = torch.tensor(idx, dtype=torch.long)

        if isinstance(out, (tuple, list)):
            return (*out, idx_t)   # <-- idx(int) 대신 idx_t(Tensor) 반환
        return out, idx_t

# ---------------------------------------------------------------------
#   PrototypeBank & Losses
# ---------------------------------------------------------------------
class PrototypeBank:
    def __init__(self, num_classes: int, dim: int, device, momentum: float = 0.9):
        self.num_classes = num_classes
        self.dim = dim
        self.momentum = momentum
        self.device = device

        self.prototypes = torch.zeros(num_classes, dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, x_cls: torch.Tensor, bag_label: torch.Tensor):
        B, C, D = x_cls.shape
        assert C == self.num_classes and D == self.dim

        for c in range(C):
            mask = bag_label[:, c] > 0
            if mask.any():
                cls_c = x_cls[mask, c, :]          # [N_c, D]
                proto_batch = cls_c.mean(dim=0)    # [D]
                if self.initialized[c]:
                    m = self.momentum
                    self.prototypes[c] = m * self.prototypes[c] + (1.0 - m) * proto_batch
                else:
                    self.prototypes[c] = proto_batch
                    self.initialized[c] = True

    def get(self):
        return self.prototypes, self.initialized

    @torch.no_grad()
    def sync(self, world_size: int):
        """
        DDP 환경에서 모든 rank의 prototype을 평균 내고,
        initialized는 OR로 합치는 동기화 함수.
        """
        if (world_size <= 1) or (not dist.is_available()) or (not dist.is_initialized()):
            return

        # 1) prototypes 평균
        dist.all_reduce(self.prototypes, op=dist.ReduceOp.SUM)
        self.prototypes /= float(world_size)

        # 2) initialized OR
        # bool은 바로 all_reduce가 안 되므로 float/int로 캐스팅
        init_float = self.initialized.to(self.prototypes.dtype)  # [C]
        dist.all_reduce(init_float, op=dist.ReduceOp.SUM)
        self.initialized = init_float > 0.0


def _densify_from_dist(dist_sub: torch.Tensor, tau: float, alpha: float):
    """
    dist_sub: [B,B] 거리 (작을수록 유사)
    returns soft assignment W: [B,B], rows sum to 1, diagonal=0
    """
    # logits = -dist / tau
    logits = -dist_sub / max(tau, 1e-6)
    # remove self
    logits.fill_diagonal_(-1e9)

    # softmax => weights
    W = torch.softmax(logits, dim=1)

    # optional sharpening: alpha>0 makes distribution peakier
    if alpha is not None and alpha > 0:
        W = W ** (1.0 + alpha)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)

    # keep diagonal 0
    W.fill_diagonal_(0.0)
    return W

def _densify_from_sim(sim_sub: torch.Tensor, tau: float, alpha: float):
    """
    sim_sub: [B,B] 유사도 (클수록 유사)
    returns soft assignment W: [B,B], rows sum to 1, diagonal=0

    - SoftCLT에서 저장한 soft_matrix(=normalized similarity)를 그대로 사용할 때 쓰는 버전
    - tau: temperature (작을수록 더 peak)
    - alpha: optional sharpening (>=0) (클수록 더 peak)
    """
    # logits = sim / tau  (similarity는 클수록 positive)
    logits = sim_sub / max(tau, 1e-6)

    # remove self
    logits.fill_diagonal_(-1e9)

    # softmax => weights
    W = torch.softmax(logits, dim=1)

    # optional sharpening: alpha>0 makes distribution peakier
    if alpha is not None and alpha > 0:
        W = W ** (1.0 + alpha)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)

    # keep diagonal 0
    W.fill_diagonal_(0.0)
    return W


def softclt_instance_loss(x_seq: torch.Tensor,
                          idx: torch.Tensor,
                          dist_mat_full_cpu: torch.Tensor,
                          tau_inst: float,
                          alpha: float):
    """
    x_seq: [B,T,D] (from your model)
    idx:   [B] dataset indices for lookup
    dist_mat_full_cpu: [N,N] on CPU
    """
    device = x_seq.device
    B, T, D = x_seq.shape

    # bag embedding (minimal intrusion): mean over time
    z = x_seq.mean(dim=1)                     # [B,D]
    z = F.normalize(z, dim=-1)

    # sub distance matrix from full (CPU -> GPU)
    idx_cpu = idx.detach().cpu()
    dist_sub = dist_mat_full_cpu.index_select(0, idx_cpu).index_select(1, idx_cpu)  # [B,B] CPU
    dist_sub = dist_sub.to(device=device, non_blocking=True)

    # W = _densify_from_dist(dist_sub, tau=tau_inst, alpha=alpha)  # [B,B]
    W = _densify_from_sim(dist_sub, tau=tau_inst, alpha=alpha)

    # logits among batch samples
    logits = torch.matmul(z, z.t()) / max(tau_inst, 1e-6)  # [B,B]
    logits.fill_diagonal_(-1e9)
    logp = F.log_softmax(logits, dim=1)                    # [B,B]

    # soft cross-entropy
    loss = -(W * logp).sum(dim=1).mean()
    return loss


def softclt_temporal_loss(x_seq: torch.Tensor,
                          tau_temp: float,
                          win: int,
                          sigma: float):
    """
    Temporal soft assignment within each sequence.
    For each time t, target distribution puts mass on [t-win, t+win] using Gaussian over |dt|.
    x_seq: [B,T,D]
    """
    device = x_seq.device
    B, T, D = x_seq.shape
    z = F.normalize(x_seq, dim=-1)                   # [B,T,D]

    # sim over time within each sample: [B,T,T]
    sim = torch.matmul(z, z.transpose(1, 2)) / max(tau_temp, 1e-6)

    # build soft targets: [T,T] same for all B
    t = torch.arange(T, device=device)
    dt = (t[None, :] - t[:, None]).abs().float()     # [T,T]
    mask = (dt <= float(win)).float()                # [T,T]

    # gaussian weights on dt
    w = torch.exp(-(dt ** 2) / (2.0 * max(sigma, 1e-6) ** 2)) * mask
    w.fill_diagonal_(0.0)

    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)     # row-normalize
    w = w.unsqueeze(0).expand(B, -1, -1)             # [B,T,T]

    # log softmax over candidate times
    sim = sim.masked_fill(torch.eye(T, device=device).bool().unsqueeze(0), -1e9)
    logp = F.log_softmax(sim, dim=2)                 # [B,T,T]

    loss = -(w * logp).sum(dim=2).mean()             # mean over (B,T)
    return loss

def instance_prototype_contrastive_loss(
    x_seq: torch.Tensor,        # [B, T, D]
    bag_label: torch.Tensor,    # [B, C]
    proto_bank: PrototypeBank,
    tau: float = 0.1,
    sim_thresh: float = 0.7,
    win: int = 5,
):
    """
    벡터화 버전:
      1) S_valid[b,t,c] = <z_{b,t}, p_c> (bag에 존재 + proto init된 class만)
      2) direct non-ambiguous: max_c S_valid[b,t,c] >= sim_thresh
      3) ambiguous: direct 실패 시, [t-win,t+win] 내에서
         max_{t'} max_c S_valid[b,t',c] >= sim_thresh 이면 그 (t',c)를 positive로 사용
    """
    device = x_seq.device
    prototypes, initialized = proto_bank.get()  # [C,D], [C]

    if not initialized.any():
        return torch.tensor(0.0, device=device)

    B, T, D = x_seq.shape
    C = prototypes.shape[0]
    assert bag_label.shape == (B, C)

    # 1. cosine normalize
    x_norm = F.normalize(x_seq, dim=-1)     # [B,T,D]
    p_norm = F.normalize(prototypes, dim=-1)   # [C,D]

    # 2. similarity S[b,t,c]
    S_full = torch.einsum('btd,cd->btc', x_norm, p_norm)  # [B,T,C]
    S_valid = S_full.clone()

    # valid class: bag_label=1 AND prototype initialized
    # valid_class_mask = initialized.view(1,1,C)
    valid_class_mask = (bag_label > 0).unsqueeze(1) & initialized.view(1, 1, C)  # [B,1,C]
    S_valid = S_valid.masked_fill(~valid_class_mask, -1e9)

    # 3. direct non-ambiguous: per (b,t)에서 class 최대
    S_tmax, c_argmax = S_valid.max(dim=-1)  # [B,T], [B,T]
    direct_pos_mask = (S_tmax >= sim_thresh)  # [B,T]

    # 4. temporal neighbor-based ambiguous matching (sliding window max over time)
    if win > 0:
        # padding 후 unfold로 윈도우 생성: [B,T,2*win+1]
        S_padded = F.pad(S_tmax, (win, win), value=-1e9)
        windows = S_padded.unfold(dimension=1, size=2 * win + 1, step=1)  # [B,T,2*win+1]

        nei_max, idx_in_win = windows.max(dim=-1)  # [B,T], [B,T]
        neighbor_pos_mask = (~direct_pos_mask) & (nei_max >= sim_thresh)

        # neighbor 시점 t' 계산
        t_arange = torch.arange(T, device=device).view(1, T)  # [1,T]
        neighbor_t = t_arange - win + idx_in_win              # [B,T]
        neighbor_t = neighbor_t.clamp(0, T - 1)

        # neighbor 시점에서의 best class
        c_nei = c_argmax.gather(1, neighbor_t)  # [B,T]
    else:
        nei_max = torch.full_like(S_tmax, -1e9)
        neighbor_pos_mask = torch.zeros_like(S_tmax, dtype=torch.bool)
        c_nei = torch.zeros_like(c_argmax)

    # 5. 최종 anchor mask 및 class 선택
    anchors_mask = direct_pos_mask | neighbor_pos_mask  # [B,T]
    if not anchors_mask.any():
        return torch.tensor(0.0, device=device)

    # 각 위치별 최종 class index (direct vs neighbor 중 선택)
    c_all = torch.where(direct_pos_mask, c_argmax, c_nei)  # [B,T]

    # (b,t,c) 인덱스를 1D 텐서로 뽑기
    b_idx_full = torch.arange(B, device=device).view(B, 1).expand(B, T)  # [B,T]
    t_idx_full = torch.arange(T, device=device).view(1, T).expand(B, T)  # [B,T]

    b_idx = b_idx_full[anchors_mask]  # [N]
    t_idx = t_idx_full[anchors_mask]  # [N]
    c_idx = c_all[anchors_mask]       # [N]
    N = b_idx.size(0)

    # 6. InfoNCE 계산
    z_anchor = x_norm[b_idx, t_idx, :]      # [N,D]

    # anchor vs all prototypes
    sim_all = torch.matmul(z_anchor, p_norm.t()) / tau   # [N,C]
    pos_sim = sim_all[torch.arange(N, device=device), c_idx]  # [N]
    log_all = torch.logsumexp(sim_all, dim=-1)                 # [N]
    base = pos_sim - log_all                                   # [N]

    # class-balanced weighting (그대로 유지)
    with torch.no_grad():
        counts = torch.bincount(c_idx, minlength=C).float()   # [C]
        weights_per_class = torch.zeros(C, device=device)
        valid = counts > 0
        weights_per_class[valid] = counts.sum() / counts[valid]

    w = weights_per_class[c_idx]   # [N]
    loss = -(base * w).sum() / (w.sum() + 1e-6)
    return loss

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

# ---------------------------------------------------------------------
#   Train / Test (DDP-aware)
# ---------------------------------------------------------------------
def train(trainloader, milnet, criterion, optimizer, epoch, args, device, proto_bank=None, is_main=True, world_size=1, dist_mat_full_cpu=None):
    milnet.train()
    total_loss = 0
    sum_bag = 0.0
    sum_inst = 0.0
    sum_ortho = 0.0
    sum_smooth = 0.0
    sum_sparsity = 0.0
    sum_ctx_contrast = 0.0
    sum_proto_inst = 0.0
    sum_cls_contrast = 0.0
    sum_total = 0.0
    n = 0

    for batch_id, batch in enumerate(trainloader):
        feats, label, idx = batch
        bag_feats = feats.to(device)
        bag_label = label.to(device)   # [B, C] multi-hot

        if args.dropout_patch > 0:
            selecy_window_indx = random.sample(range(10), int(args.dropout_patch * 10))
            inteval = int(len(bag_feats) // 10)
            for pidx in selecy_window_indx:
                bag_feats[:, pidx * inteval:pidx * inteval + inteval, :] = torch.randn(1, device=device)

        optimizer.zero_grad()

        # forward
        if args.model == 'AmbiguousMIL':
            bag_prediction, instance_pred, weighted_instance_pred, non_weighted_instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(
                bag_feats
            )
        else:
            if epoch < args.epoch_des:
                bag_prediction, x_cls, attn_layer1, attn_layer2 = milnet(bag_feats, warmup=True)
                instance_pred = None
            else:
                bag_prediction, x_cls, attn_layer1, attn_layer2 = milnet(bag_feats, warmup=False)
                instance_pred = None

        # bag-level loss
        bag_loss = criterion(bag_prediction, bag_label)

        # 초기값 세팅
        inst_loss = 0.0
        ortho_loss = torch.tensor(0.0, device=device)
        smooth_loss = torch.tensor(0.0, device=device)
        sparsity_loss = torch.tensor(0.0, device=device)
        soft_aux_loss = torch.tensor(0.0, device=device)
        if instance_pred is not None:
            inst_loss = criterion(instance_pred, bag_label)

        if args.use_softclt_aux and (epoch >= args.epoch_des):
            # SoftCLT auxiliary losses (instance-wise + temporal)
            # x_seq is required; AmbiguousMILwithCL already returns x_seq   
            # If your model path can produce x_seq only when instance_pred exists,
            # this block naturally activates only in that regime.
            L_inst = softclt_instance_loss(
                x_seq=x_seq,
                idx=idx.to(device=device),
                dist_mat_full_cpu=dist_mat_full_cpu,
                tau_inst=args.soft_tau_inst,
                alpha=args.soft_alpha,
            )
            L_temp = softclt_temporal_loss(
                x_seq=x_seq,
                tau_temp=args.soft_tau_temp,
                win=args.soft_temp_win,
                sigma=args.soft_temp_sigma,
            )
            soft_aux_loss = args.soft_lambda * L_inst + (1.0 - args.soft_lambda) * L_temp

        # 최종 loss
        if args.model == 'AmbiguousMIL':
            loss = (
                args.bag_loss_w * bag_loss
                + args.inst_loss_w * inst_loss
                # + args.ortho_loss_w * ortho_loss
                # + args.smooth_loss_w * smooth_loss
                + args.sparsity_loss_w * sparsity_loss
                + args.proto_loss_w * soft_aux_loss
                # + args.cls_contrast_w * cls_contrast_loss
            )
        else:
            loss = bag_loss

        if is_main:
            sys.stdout.write(
                '\r [Train] Epoch %d | bag [%d/%d] bag loss: %.4f  total loss: %.4f' %
                (epoch, batch_id, len(trainloader), bag_loss.item(), loss.item())
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()

        # 통계값 누적
        total_loss += bag_loss.item()
        sum_bag += bag_loss.item()
        sum_inst += float(inst_loss) if isinstance(inst_loss, float) else float(inst_loss)
        # sum_ortho += ortho_loss.item()
        # sum_smooth += smooth_loss.item()
        sum_sparsity += sparsity_loss.item()
        # sum_proto_inst += proto_inst_loss.item()
        sum_proto_inst += soft_aux_loss.item()
        # sum_cls_contrast += cls_contrast_loss.item()
        sum_total += loss.item()
        n += 1

    if is_main and wandb.run is not None:
        wandb.log({
            "epoch": epoch,
            "train/bag_loss": sum_bag / max(1, n),
            "train/inst_loss": sum_inst / max(1, n),
            # "train/ortho_loss": sum_ortho / max(1, n),
            # "train/smooth_loss": sum_smooth / max(1, n),
            "train/sparsity_loss": sum_sparsity / max(1, n),
            # "train/ctx_contrast_loss": sum_ctx_contrast / max(1, n),
            "train/proto_inst_loss": sum_proto_inst / max(1, n),
            # "train/cls_contrast_loss": sum_cls_contrast / max(1, n),
            "train/total_loss": sum_total / max(1, n),
        }, step=epoch)

    return total_loss / max(1, n)


def test(testloader, milnet, criterion, epoch, args, device, threshold: float = 0.5, proto_bank=None, is_main=True):
    if isinstance(milnet, nn.parallel.DistributedDataParallel):
        model = milnet.module
    else:
        model = milnet
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    sum_bag = 0.0
    sum_inst = 0.0
    sum_ortho = 0.0
    sum_smooth = 0.0
    sum_sparsity = 0.0
    sum_ctx_contrast = 0.0
    sum_proto_inst = 0.0
    sum_cls_contrast = 0.0
    sum_total = 0.0
    n = 0

    inst_total_correct = 0
    inst_total_count   = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(testloader):

            if args.datatype == "mixed":
                feats, label, y_inst, idx = batch
            else:
                feats, label, idx = batch
            bag_feats = feats.to(device)
            bag_label = label.to(device)   # [B, C]

            if args.model == 'AmbiguousMIL':
                bag_prediction, instance_pred, weighted_instance_pred, non_weighted_instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(bag_feats)
            elif args.model == 'newTimeMIL':
                out = model(bag_feats)
                instance_pred = None
                attn_layer2 = None
                if isinstance(out, (tuple, list)):
                    bag_prediction = out[0]
                    attn_layer2 = out[3]
                else:
                    bag_prediction = out
            elif args.model == 'TimeMIL':
                out = model(bag_feats)
                instance_pred = None
                attn_layer2 = None
                if isinstance(out, (tuple, list)):
                    bag_prediction = out[0]
                    attn_layer2 = out[2]
                else:
                    bag_prediction = out
            else:
                raise ValueError(f"Unknown model name: {args.model}")

            # bag-level loss
            bag_loss = criterion(bag_prediction, bag_label)

            inst_loss = 0.0
            # ortho_loss = torch.tensor(0.0, device=device)
            # smooth_loss = torch.tensor(0.0, device=device)
            sparsity_loss = torch.tensor(0.0, device=device)
            proto_inst_loss = torch.tensor(0.0, device=device)
            # cls_contrast_loss = torch.tensor(0.0, device=device)

            if instance_pred is not None:
                # # instance → bag MIL loss
                # p_inst = torch.sigmoid(instance_pred)        # [B, T, C]
                # eps = 1e-6
                # p_bag_from_inst = 1 - torch.prod(
                #     torch.clamp(1 - p_inst, min=eps), dim=1
                # )                                           # [B, C]
                # inst_loss = F.binary_cross_entropy(
                #     p_bag_from_inst, bag_label.float()
                # )

                # p_inst = instance_pred.mean(dim=1)       # [B, C]
                inst_loss = criterion(instance_pred, bag_label)

                # # class token orthogonality
                # ortho_loss = class_token_orthogonality_loss(x_cls)

                # 이번 배치에서 실제로 등장한 positive class만 선택
                pos_mask = (bag_label.sum(dim=0) > 0).float()   # [C]

                # sparsity loss
                instance_pred_s = torch.sigmoid(weighted_instance_pred)
                p_inst_s = instance_pred_s.mean(dim=1)
                sparsity_per_class = p_inst_s.mean(dim=(0, 1))    # [C]
                # if pos_mask.sum() > 0:
                #     sparsity_loss = (sparsity_per_class * pos_mask).sum() / (
                #         pos_mask.sum() + 1e-6
                #     )
                # else:
                    # sparsity_loss = torch.tensor(0.0, device=device)
                sparsity_loss = sparsity_per_class.mean()   # pos_mask 제거 버전

                # # temporal smoothness loss
                # diff = p_inst[:, 1:, :] - p_inst[:, :-1, :]   # [B, T-1, C]
                # diff = diff * pos_mask[None, None, :]         # broadcast
                # smooth_loss = (diff ** 2).mean()

                # prototype 기반 instance CL (eval에서는 update 없이 loss만)
                # if (epoch >= args.epoch_des) and (proto_bank is not None):
                #     proto_inst_loss = instance_prototype_contrastive_loss(
                #         x_seq,
                #         bag_label,
                #         proto_bank,
                #         tau=getattr(args, "proto_tau", 0.1),
                #         sim_thresh=getattr(args, "proto_sim_thresh", 0.7),
                #         win=getattr(args, "proto_win", 5),
                #     )

            if args.datatype == "mixed" and (y_inst is not None):
                # y_inst: [B, T, C] one-hot instance labels
                # CPU에 있어도 되지만, 일단 계산 편의를 위해 device로 올림
                y_inst = y_inst.to(device)
                # timestep별 정답 클래스 인덱스
                y_inst_label = torch.argmax(y_inst, dim=2)  # [B, T]

                # AmbiguousMIL인 경우 instance_pred에서 직접 argmax
                if args.model == 'AmbiguousMIL':
                    pred_inst = torch.argmax(weighted_instance_pred, dim=2)  # [B, T]
                elif args.model == 'newTimeMIL' and attn_layer2 is not None:
                    B, T, C = y_inst.shape
                    attn_cls = attn_layer2[:,:,:C,C:]
                    attn_mean = attn_cls.mean(dim=1)
                    pred_inst = torch.argmax(attn_mean, dim=1)
                else:
                    pred_inst = None

                if pred_inst is not None:
                    correct = (pred_inst == y_inst_label).sum().item()
                    count   = y_inst_label.numel()
                    inst_total_correct += correct
                    inst_total_count   += count

            if args.model == 'AmbiguousMIL':
                loss = (
                    args.bag_loss_w * bag_loss
                    + args.inst_loss_w * inst_loss
                    # + args.ortho_loss_w * ortho_loss
                    # + args.smooth_loss_w * smooth_loss
                    + args.sparsity_loss_w * sparsity_loss
                    + args.proto_loss_w * proto_inst_loss
                    # + args.cls_contrast_w * cls_contrast_loss
                )
            else:
                loss = bag_loss

            if is_main:
                sys.stdout.write(
                    '\r [Val]   Epoch %d | bag [%d/%d] bag loss: %.4f  total loss: %.4f' %
                    (epoch, batch_id, len(testloader), bag_loss.item(), loss.item())
                )

            total_loss += loss.item()
            
            if args.model == 'AmbiguousMIL':
                probs = torch.sigmoid(instance_pred).cpu().numpy() # p_inst is main output for evaluation
            else:
                probs = torch.sigmoid(bag_prediction).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

            sum_bag += bag_loss.item()
            sum_inst += float(inst_loss) if isinstance(inst_loss, float) else float(inst_loss)
            # sum_ortho += ortho_loss.item()
            # sum_smooth += smooth_loss.item()
            sum_sparsity += sparsity_loss.item()
            sum_proto_inst += proto_inst_loss.item()
            # sum_cls_contrast += cls_contrast_loss.item()
            sum_total += loss.item()
            n += 1

    # metric 계산
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= threshold).astype(np.int32)

    inst_acc = None
    if inst_total_count > 0:
        inst_acc = float(inst_total_correct) / float(inst_total_count)

    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    p_micro  = precision_score(y_true, y_pred, average='micro', zero_division=0)
    p_macro  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r_micro  = recall_score(y_true, y_pred, average='micro', zero_division=0)
    r_macro  = recall_score(y_true, y_pred, average='macro', zero_division=0)

    roc_list, ap_list = [], []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) == 2:
            try:
                roc_list.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
                ap_list.append(average_precision_score(y_true[:, c], y_prob[:, c]))
            except Exception:
                pass

    roc_macro = float(np.mean(roc_list)) if roc_list else 0.0
    ap_macro  = float(np.mean(ap_list))  if ap_list  else 0.0

    bag_acc = None
    if args.datatype == "original":
        # y_true 가 one-hot (또는 multi-hot인데 실제로는 한 클래스만 1)이라고 가정
        true_cls = y_true.argmax(axis=1)
        pred_cls = y_prob.argmax(axis=1)
        bag_acc = float((true_cls == pred_cls).mean())

    if is_main and wandb.run is not None:
        log_dict = {
            "val/bag_loss": sum_bag / max(1, n),
            "val/inst_loss": sum_inst / max(1, n),
            # "val/ortho_loss": sum_ortho / max(1, n),
            # "val/smooth_loss": sum_smooth / max(1, n),
            "val/sparsity_loss": sum_sparsity / max(1, n),
            # "val/ctx_contrast_loss": sum_ctx_contrast / max(1, n),
            "val/proto_inst_loss": sum_proto_inst / max(1, n),
            # "val/cls_contrast_loss": sum_cls_contrast / max(1, n),
            "val/total_loss": sum_total / max(1, n),
        }
        if bag_acc is not None:
            log_dict["val/bag_acc"] = bag_acc
        wandb.log(log_dict, step=epoch)

    results = {
        "f1_micro": f1_micro, "f1_macro": f1_macro,
        "p_micro": p_micro,   "p_macro": p_macro,
        "r_micro": r_micro,   "r_macro": r_macro,
        "roc_auc_macro": roc_macro, "mAP_macro": ap_macro
    }
    if bag_acc is not None:
        results["bag_acc"] = bag_acc
    
    if inst_acc is not None:
        results["inst_acc"] = inst_acc

    return total_loss / max(1, n), results

# ---------------------------------------------------------------------
#   main (DDP 진입점)
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="PenDigits", type=str, help='dataset')
    parser.add_argument('--datatype', default="mixed", type=str, help='Choose datatype between original and mixed')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512] resnet-50 1024')
    parser.add_argument('--lr', default=5e-3, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='(단일 GPU 실행 시 사용 가능, DDP에서는 무시 가능)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.5, type=float, help='Patch dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='TimeMIL', type=str, help='MIL model')
    parser.add_argument('--prepared_npz', type=str, default='./data/PAMAP2.npz', help='전처리 결과 npz 경로 (예: ./data/PAMAP2.npz)')
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')

    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=20, type=int, help='turn on warmup')

    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    parser.add_argument('--ctx_win', type=int, default=4,
                        help='context window radius for contrastive loss')
    parser.add_argument('--ctx_tau', type=float, default=0.1,
                        help='temperature for context contrastive loss')

    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--proto_tau', type=float, default=0.1,
                        help='temperature for instance-prototype contrastive loss')
    parser.add_argument('--proto_sim_thresh', type=float, default=0.5,
                        help='similarity threshold to treat instance as confident')
    parser.add_argument('--proto_win', type=int, default=5,
                        help='temporal neighbor window for ambiguous instances')
    # loss weights
    parser.add_argument('--bag_loss_w', type=float, default=0.5, help='weight for bag-level loss')
    parser.add_argument('--inst_loss_w', type=float, default=0.2, help='weight for instance-level MIL loss')
    parser.add_argument('--ortho_loss_w', type=float, default=0.0, help='weight for class token orthogonality loss')
    parser.add_argument('--smooth_loss_w', type=float, default=0.05, help='weight for temporal smoothness loss')
    parser.add_argument('--sparsity_loss_w', type=float, default=0.05, help='weight for sparsity loss')
    parser.add_argument('--ctx_contrast_w', type=float, default=0.0, help='weight for context contrastive loss')
    parser.add_argument('--proto_loss_w', type=float, default=0.2, help='weight for prototype instance contrastive loss')
    parser.add_argument('--cls_contrast_tau', type=float, default=0.1, help='temperature for batch class embedding contrastive loss')
    parser.add_argument('--cls_contrast_w', type=float, default=0.0, help='weight for batch class embedding contrastive loss')
    
    parser.add_argument('--dba_root', type=str, default='./data/dba_data', help='Root directory of DBA dataset (1_xxx/aggressive/... 구조)')
    parser.add_argument('--dba_window', type=int, default=12000, help='Sliding window length (time steps) for DBA dataset')
    parser.add_argument('--dba_stride', type=int, default=6000, help='Sliding window stride for DBA dataset')
    parser.add_argument('--dba_test_ratio', type=float, default=0.2, help='Train/Test split ratio (sequence-level) for DBA dataset')

    # --- SoftCLT auxiliary losses (replace proto_inst_loss) ---
    parser.add_argument('--use_softclt_aux', action='store_true',
                        help='Use SoftCLT (instance-wise + temporal) auxiliary losses instead of proto_inst_loss')
    parser.add_argument('--sim_mat_path', type=str, default='',
                        help='Path to precomputed similarity matrix for TRAINSET indices (npz with key "sim_mat" or "dist_mat")')
    parser.add_argument('--soft_tau_inst', type=float, default=0.1,
                        help='temperature for instance-wise soft contrastive')
    parser.add_argument('--soft_tau_temp', type=float, default=0.1,
                        help='temperature for temporal soft contrastive')
    parser.add_argument('--soft_alpha', type=float, default=0.5,
                        help='row sharpening for soft assignment (>=0); higher = peakier')
    parser.add_argument('--soft_lambda', type=float, default=0.5,
                        help='mixing weight: lambda*L_inst + (1-lambda)*L_temp')
    parser.add_argument('--soft_temp_win', type=int, default=5,
                        help='temporal soft positive window radius')
    parser.add_argument('--soft_temp_sigma', type=float, default=2.0,
                        help='gaussian sigma (in timesteps) for temporal soft target distribution')
    
    parser.add_argument('--build_dist_mat', action='store_true',
                        help='If set, build dist_mat_full from TRAINSET and save to --dist_out then exit.')
    parser.add_argument('--dist_out', type=str, default='./dist_mat_train.npz',
                        help='Output .npz path for dist_mat_full')
    parser.add_argument('--dist_metric', type=str, default='DTW',
                        choices=['DTW','TAM','GAK','COS','EUC'],
                        help='Metric for building similarity matrix (SoftCLT style)')

    parser.add_argument('--dist_min', type=float, default=0.0, help='Min for MinMaxScaler')
    parser.add_argument('--dist_max', type=float, default=1.0, help='Max for MinMaxScaler')
    parser.add_argument('--dist_max_items', type=int, default=0,
                        help='If >0, compute dist matrix only for first K items (debug)')
    
    parser.add_argument('--densify_tau', type=float, default=None)
    parser.add_argument('--densify_alpha', type=float, default=None)
    
    args = parser.parse_args()

    # ----- DDP setup -----
    rank, world_size, local_rank, is_main = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # rank별 seed 다르게
    base_seed = args.seed
    random.seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    torch.manual_seed(base_seed + rank)
    torch.cuda.manual_seed_all(base_seed + rank)

    # ----- 디렉토리 / wandb / logger -----
    args.save_dir = args.save_dir + 'InceptBackbone'
    dataset_root = join(args.save_dir, f'{args.dataset}')
    maybe_mkdir_p(dataset_root)

    # rank 0만 새 exp 폴더를 만들고 경로를 브로드캐스트해서 파일 의존성을 없앤다.
    if is_main:
        exp_path = make_dirs(dataset_root)   # 예: ./savemodel/InceptBackbone/ArticularyWordRecognition/exp_0
    else:
        exp_path = None

    if world_size > 1:
        # broadcast exp_path to all ranks so everyone uses the same directory
        exp_path_list = [exp_path]
        dist.broadcast_object_list(exp_path_list, src=0)
        exp_path = exp_path_list[0]
        dist.barrier()  # ensure path is shared before proceeding

    args.save_dir = exp_path
    maybe_mkdir_p(args.save_dir)

    version_name = os.path.basename(exp_path)

    if is_main:
        print(f'Running TimeMIL on dataset: {args.dataset}')
        wandb.init(project="TimeMIL", name=f"{args.dataset}_{args.model}_{version_name}", config=vars(args))
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")
        wandb.define_metric("score/*", step_metric="epoch")
        logging_path = os.path.join(args.save_dir, 'Train_log.log')
        logger = get_logger(logging_path)
    else:
        logger = None

    # loss weight 설정 (이미 parser에서 주입됨)

    if is_main:
        option = vars(args)
        file_name = os.path.join(args.save_dir, 'option.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(option.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    criterion = nn.BCEWithLogitsLoss()

    def _dataset_to_X_yidx(ds, device="cpu"):
        """
        loadorean()이 만든 Dataset에서 전체를 꺼내
        X: [N, T, D] float tensor
        y_idx: [N] long tensor (0..C-1)
        로 변환합니다.

        ds[i]가 (x, y) 또는 (x, y, ...) 형태여도 첫 2개만 사용합니다.
        y가 one-hot이면 argmax로 idx를 만듭니다.
        """
        X_list = []
        y_list = []

        for i in range(len(ds)):
            item = ds[i]
            if isinstance(item, (tuple, list)):
                x = item[0]
                y = item[1]
            else:
                raise ValueError("Dataset item must be tuple/list like (x, y)")

            # x: [T, D] or [D, T] 등 가능성이 있으면 여기서 정규화
            # 현재 코드 흐름 상 loadorean은 보통 [T, D]를 준다고 가정합니다.
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            x = x.float()

            if not torch.is_tensor(y):
                y = torch.tensor(y)

            # y가 one-hot([C]) 또는 scalar일 수 있음
            if y.ndim > 0 and y.numel() > 1:
                y_idx = int(torch.argmax(y).item())
            else:
                y_idx = int(y.item())

            X_list.append(x)
            y_list.append(y_idx)

        X = torch.stack(X_list, dim=0).to(device)                 # [N, T, D]
        y_idx = torch.tensor(y_list, dtype=torch.long, device=device)  # [N]
        return X, y_idx

    # ---------------- 데이터 구성 ----------------
    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:
        train_base = loadorean(args, split='train')
        test_base  = loadorean(args, split='test')

        base_T = train_base.max_len
        num_classes = train_base.num_class
        L_in = train_base.feat_in

        if is_main:
            print(f"[{args.dataset}] base max_len={base_T}, num_classes={num_classes}, feat_in={L_in}")

        # 공통 args 갱신(입력 피처/클래스 수)
        args.feats_size = L_in
        args.num_classes = num_classes

        if args.datatype == "original":
            # 기존대로 사용
            trainset = train_base
            testset  = test_base
            seq_len  = base_T
            args.seq_len = seq_len

        elif args.datatype == "mixed":
            # loadorean dataset -> (X, y_idx)로 변환 후 mixed bag으로 래핑
            # 주의: mixed bag은 길이가 concat_k * T가 되므로 seq_len도 그에 맞춰야 합니다.
            Xtr, ytr_idx = _dataset_to_X_yidx(train_base, device="cpu")
            Xte, yte_idx = _dataset_to_X_yidx(test_base,  device="cpu")

            concat_k = 2  # 원하시면 args로 빼셔도 됩니다(예: --concat_k)
            trainset = MixedSyntheticBagsConcatK(
                X=Xtr,
                y_idx=ytr_idx,
                num_classes=num_classes,
                total_bags=len(Xtr),      # 필요시 더 크게(예: 5000)도 가능
                concat_k=concat_k,
                seed=args.seed,
                return_instance_labels=False,
            )
            testset = MixedSyntheticBagsConcatK(
                X=Xte,
                y_idx=yte_idx,
                num_classes=num_classes,
                total_bags=len(Xte),
                concat_k=concat_k,
                seed=args.seed + 1,
                return_instance_labels=True,   # mixed eval에서 inst acc 쓰려면 True
            )

            seq_len = concat_k * base_T
            args.seq_len = seq_len

            if is_main:
                print(f"[{args.dataset}] mixed concat_k={concat_k} -> seq_len={seq_len}")

        else:
            raise ValueError(f"Unsupported datatype '{args.datatype}' for {args.dataset}")

    elif args.dataset in ['PAMAP2']:
        trainset = loadorean(args, split='train')
        testset  = loadorean(args, split='test')
        seq_len, num_classes, L_in = trainset.max_len, trainset.num_class, trainset.feat_in

        args.seq_len = seq_len
        if is_main:
            print(f'max length {seq_len}')
        args.feats_size  = L_in
        args.num_classes = num_classes
        if is_main:
            print(f'num class:{args.num_classes}')
    elif args.dataset == 'dba':
        if is_main:
            print(f"[DBA] building dataset from root: {args.dba_root}")

        if args.datatype == 'original':
            # window 하나가 bag이 되는 기존 방식
            trainset, testset, seq_len, num_classes, L_in = build_dba_for_timemil(args)

        elif args.datatype == 'mixed':
            # window 단위로 잘라서 concat-bag 을 만드는 방식
            Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, L_in = build_dba_windows_for_mixed(args)

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
                seed=args.seed+1,
                return_instance_labels=True,
            )

        else:
            raise ValueError(f"Unsupported datatype '{args.datatype}' for DBA dataset")

        args.seq_len = seq_len
        args.feats_size = L_in
        args.num_classes = num_classes

        if is_main:
            print(f"[DBA] seq_len (window): {seq_len}")
            print(f"[DBA] num_classes: {num_classes}")
            print(f"[DBA] feat_in: {L_in}")
    else:
        Xtr, ytr, meta = load_classification(name=args.dataset, split='train',extract_path='./data')
        Xte, yte, _   = load_classification(name=args.dataset, split='test',extract_path='./data')

        word_to_idx = {cls:i for i, cls in enumerate(meta['class_values'])}
        ytr_idx = torch.tensor([word_to_idx[i] for i in ytr], dtype=torch.long)
        yte_idx = torch.tensor([word_to_idx[i] for i in yte], dtype=torch.long)

        Xtr = torch.from_numpy(Xtr).permute(0,2,1).float()
        Xte = torch.from_numpy(Xte).permute(0,2,1).float()

        num_classes = len(meta['class_values'])
        args.num_classes = num_classes
        L_in = Xtr.shape[-1]
        seq_len = max(21, Xte.shape[1])

        mix_probs = {'orig': 0.4, 2: 0.4, 3: 0.2}
        SYN_TOTAL_LEN = seq_len

        if args.datatype == 'mixed':
            trainset = MixedSyntheticBagsConcatK(
                X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
                total_bags=len(Xtr),
                seed=args.seed
            )

            testset = MixedSyntheticBagsConcatK(
                X=Xte, y_idx=yte_idx, num_classes=num_classes,
                total_bags=len(Xte),
                seed=args.seed+1,
                return_instance_labels=True
            )
        elif args.datatype == 'original':
            # ★ 여기서는 한 bag에 하나의 class (원본 시퀀스)
            trainset = TensorDataset(Xtr, F.one_hot(ytr_idx, num_classes=num_classes).float())
            testset  = TensorDataset(Xte, F.one_hot(yte_idx, num_classes=num_classes).float())

        args.seq_len = seq_len
        args.feats_size = L_in
        
        if is_main:
            print(f'num class: {args.num_classes}')
            print(f'total_len: {SYN_TOTAL_LEN}')
    
    trainset = WithIndexDataset(trainset)
    testset  = WithIndexDataset(testset)

    if args.build_dist_mat:
        if is_main:
            build_softclt_soft_matrix_from_dataset(
                trainset,
                type_=args.dist_metric,
                min_=args.dist_min,
                max_=args.dist_max,
                multivariate=True,  # 필요 시 argparse로 받도록
                densify_tau=args.densify_tau,
                densify_alpha=args.densify_alpha,
                out_path=args.dist_out,
            )
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        if is_main and wandb.run is not None:
            wandb.finish()
        cleanup_distributed()
        return
    # ---------------- 모델 구성 ----------------
    if args.model =='TimeMIL':
        base_model = TimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).to(device)
        proto_bank = None
    elif args.model == 'newTimeMIL':
        base_model = TimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len, is_instance=True).to(device)
        proto_bank = None
    elif args.model == 'AmbiguousMIL':
        base_model = AmbiguousMILwithCL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, is_instance=True).to(device)
        proto_bank = PrototypeBank(
            num_classes=num_classes,
            dim=args.embed,
            device=device,
            momentum=0.9,
        )
    else:
        raise Exception("Model not available")

    if world_size > 1:
        milnet = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True  # warmup 경로 때문에
        )
    else:
        milnet = base_model

    # ---------------- Optimizer ----------------
    if  args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
        optimizer = Lookahead(optimizer, alpha=0.5, k=5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = Lookahead(optimizer, alpha=0.5, k=5)

    # ---------------- Batch size ----------------
    if args.dataset in ['DuckDuckGeese','PenDigits','FingerMovements','ERing','EigenWorms','HandMovementDirection','RacketSports','UWaveGestureLibrary']:
        batch = 64
    elif args.dataset in ['Heartbeat']:
        batch = 32
    elif args.dataset in ['EthanolConcentration','NATOPS','JapaneseVowels','MotorImagery','SelfRegulationSCP1']:
        batch = 16
    elif args.dataset in ['PEMS-SF','SelfRegulationSCP2','AtrialFibrillation','Cricket']:
        batch = 8
    elif args.dataset in ['StandWalkJump']:
        batch = 1
    elif args.dataset in ['Libras','Handwriting','Epilepsy']:
        batch = 128
    elif args.dataset in ['FaceDetection','LSST','ArticularyWordRecognition','BasicMotions','PhonemeSpectra']:
        batch = 512
    else:
        batch = args.batchsize

    # ---------------- Sampler / Dataloader ----------------
    if world_size > 1:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        # ★ val은 rank 0에서만 수행하므로 sampler=None으로 전체를 그대로 사용
        test_sampler  = None
        shuffle_train = False
    else:
        train_sampler = None
        test_sampler  = None
        shuffle_train = True

    trainloader = DataLoader(
        trainset, batch_size=batch, shuffle=shuffle_train,
        num_workers=args.num_workers, drop_last=False, pin_memory=True,
        sampler=train_sampler
    )
    testloader = DataLoader(
        testset, batch_size=batch, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True,
        sampler=test_sampler
    )

    sim_mat_full_cpu = None
    if args.use_softclt_aux:
        assert args.sim_mat_path, "--use_softclt_aux 인 경우 --sim_mat_path가 필요합니다."

        sim_mat_full_cpu = prepare_softclt_sim_matrix(
            sim_mat_path=args.sim_mat_path,
            trainset=trainset,
            args=args,
            is_main=is_main,
            world_size=world_size,
        )

    def _four_sig(val: float):
        """Return value rounded to 4 significant figures (as float) for tie checks."""
        return float(f"{val:.4g}")

    def _is_better(new_res, best_res):
        """
        Compare results with priority:
        1) primary metric (mAP or acc) by 4 sf
        2) F1 macro by 4 sf
        3) instance accuracy by 4 sf (if both exist)
        """
        primary_key = "bag_acc" if args.datatype == "original" else "mAP_macro"
        new_primary = new_res.get(primary_key, 0.0)
        best_primary = best_res.get(primary_key, -float("inf"))

        if _four_sig(new_primary) != _four_sig(best_primary):
            return new_primary > best_primary

        new_f1 = new_res.get("f1_macro", 0.0)
        best_f1 = best_res.get("f1_macro", -float("inf"))
        if _four_sig(new_f1) != _four_sig(best_f1):
            return new_f1 > best_f1

        new_inst = new_res.get("inst_acc", None)
        best_inst = best_res.get("inst_acc", None)
        if new_inst is not None and best_inst is not None:
            if _four_sig(new_inst) != _four_sig(best_inst):
                return new_inst > best_inst

        return False

    # ---------------- Training loop ----------------
    best_score = 0
    best_model_path = None
    save_path = join(args.save_dir, 'weights')
    if is_main:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(join(args.save_dir,'lesion'), exist_ok=True)
    # DDP에서 디렉토리 생성 sync
    if world_size > 1:
        dist.barrier()

    results_best = None

    for epoch in range(1, args.num_epochs + 1):
        if isinstance(trainloader.sampler, DistributedSampler):
            trainloader.sampler.set_epoch(epoch)

        train_loss_bag = train(
            trainloader, milnet, criterion, optimizer, epoch, args, device,
            dist_mat_full_cpu=sim_mat_full_cpu,
            is_main=is_main, world_size=world_size
        )

        # validation은 rank 0에서만 수행
        if is_main:
            test_loss_bag, results = test(testloader, milnet, criterion, epoch, args, device, threshold=0.5, proto_bank=proto_bank, is_main=is_main)

            if wandb.run is not None:
                log_score = {
                    "epoch": epoch,
                    "score/f1_micro": results["f1_micro"],
                    "score/f1_macro": results["f1_macro"],
                    "score/precision_micro": results["p_micro"],
                    "score/precision_macro": results["p_macro"],
                    "score/recall_micro": results["r_micro"],
                    "score/recall_macro": results["r_macro"],
                    "score/roc_auc_macro": results["roc_auc_macro"],
                    "score/mAP_macro": results["mAP_macro"]
                }
                if "bag_acc" in results:
                    log_score["score/acc"] = results["bag_acc"]
                if "inst_acc" in results:
                    log_score["score/inst_acc"] = results["inst_acc"]
                wandb.log(log_score, step=epoch)

            if logger is not None:
                if "bag_acc" in results and "inst_acc" in results:
                    logger.info(
                        ('Epoch [%d/%d] train loss: %.4f test loss: %.4f | '
                        'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                        'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f  Acc=%.4f, Inst_Acc=%.4f') %
                        (epoch, args.num_epochs, train_loss_bag, test_loss_bag,
                        results["f1_micro"], results["f1_macro"],
                        results["p_micro"],  results["p_macro"],
                        results["r_micro"],  results["r_macro"],
                        results["roc_auc_macro"], results["mAP_macro"],
                        results["bag_acc"], results["inst_acc"])
                    )
                elif "inst_acc" in results:
                    logger.info(
                        ('Epoch [%d/%d] train loss: %.4f test loss: %.4f | '
                        'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                        'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f  Inst_Acc=%.4f') %
                        (epoch, args.num_epochs, train_loss_bag, test_loss_bag,
                        results["f1_micro"], results["f1_macro"],
                        results["p_micro"],  results["p_macro"],
                        results["r_micro"],  results["r_macro"],
                        results["roc_auc_macro"], results["mAP_macro"],
                        results["inst_acc"])
                    )
                elif "bag_acc" in results:
                    logger.info(
                        ('Epoch [%d/%d] train loss: %.4f test loss: %.4f | '
                        'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                        'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f  Acc=%.4f') %
                        (epoch, args.num_epochs, train_loss_bag, test_loss_bag,
                        results["f1_micro"], results["f1_macro"],
                        results["p_micro"],  results["p_macro"],
                        results["r_micro"],  results["r_macro"],
                        results["roc_auc_macro"], results["mAP_macro"],
                        results["bag_acc"])
                    )
                else:
                    logger.info(
                        ('Epoch [%d/%d] train loss: %.4f test loss: %.4f | '
                        'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                        'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f') %
                        (epoch, args.num_epochs, train_loss_bag, test_loss_bag,
                        results["f1_micro"], results["f1_macro"],
                        results["p_micro"],  results["p_macro"],
                        results["r_micro"],  results["r_macro"],
                        results["roc_auc_macro"], results["mAP_macro"])
                    )

            should_save = False
            if results_best is None:
                should_save = True
            else:
                should_save = _is_better(results, results_best)

            if should_save:
                results_best = copy.deepcopy(results)
                primary_key = "bag_acc" if args.datatype == "original" else "mAP_macro"
                best_score = results_best.get(primary_key, 0.0)
                print(best_score)
                save_name = os.path.join(save_path, f'best_{args.model}.pth')
                best_model_path = save_name

                # DDP일 때 state_dict 구조 처리
                if isinstance(milnet, nn.parallel.DistributedDataParallel):
                    torch.save(milnet.module.state_dict(), save_name)
                else:
                    torch.save(milnet.state_dict(), save_name)

                if logger is not None:
                    logger.info('Best model saved at: ' + save_name)

        # if world_size > 1:
        #     dist.barrier()

    if is_main and results_best is not None and logger is not None:
        best = results_best
        if "bag_acc" in best:
            logger.info(
                ('Best Results | '
                'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f  Acc=%.4f') %
                (best["f1_micro"], best["f1_macro"],
                best["p_micro"],  best["p_macro"],
                best["r_micro"],  best["r_macro"],
                best["roc_auc_macro"], best["mAP_macro"],
                best["bag_acc"])
            )
        elif "inst_acc" in best:
            logger.info(
                ('Best Results | '
                'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f  Inst_Acc=%.4f') %
                (best["f1_micro"], best["f1_macro"],
                best["p_micro"],  best["p_macro"],
                best["r_micro"],  best["r_macro"],
                best["roc_auc_macro"], best["mAP_macro"],
                best["inst_acc"])
            )
        else:
            logger.info(
                ('Best Results | '
                'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f') %
                (best["f1_micro"], best["f1_macro"],
                best["p_micro"],  best["p_macro"],
                best["r_micro"],  best["r_macro"],
                best["roc_auc_macro"], best["mAP_macro"])
            )

    # ----- Final evaluation on best model: classification + AOPCR -----
    if is_main and best_model_path is not None:
        if logger is not None:
            logger.info(f"Loading best model for final eval: {best_model_path}")

        # rebuild evaluation model with is_instance=True when available
        if args.model == 'TimeMIL':
            eval_model = TimeMIL(args.feats_size, mDim=args.embed, n_classes=args.num_classes,
                                 dropout=args.dropout_node, max_seq_len=args.seq_len, is_instance=True).to(device)
        elif args.model == 'newTimeMIL':
            eval_model = TimeMIL(args.feats_size, mDim=args.embed, n_classes=args.num_classes,
                                    dropout=args.dropout_node, max_seq_len=args.seq_len, is_instance=True).to(device)
        elif args.model == 'AmbiguousMIL':
            eval_model = AmbiguousMILwithCL(args.feats_size, mDim=args.embed, n_classes=args.num_classes,
                                            dropout=args.dropout_node, is_instance=True).to(device)
        else:
            eval_model = None

        if eval_model is not None:
            state = torch.load(best_model_path, map_location=device)
            eval_model.load_state_dict(state)
            eval_model.eval()

            # Reuse testloader for classification metrics/instance accuracy
            test_loss_final, results_final = test(testloader, eval_model, criterion, epoch=args.num_epochs,
                                                  args=args, device=device, threshold=0.5,
                                                  proto_bank=None, is_main=is_main)
            if logger is not None:
                if "inst_acc" in results_final:
                    logger.info(
                        ('Final eval | '
                        'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                        'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f  Inst_Acc=%.4f') %
                        (results_final.get("f1_micro", 0.0), results_final.get("f1_macro", 0.0),
                         results_final.get("p_micro", 0.0),  results_final.get("p_macro", 0.0),
                         results_final.get("r_micro", 0.0),  results_final.get("r_macro", 0.0),
                         results_final.get("roc_auc_macro", 0.0), results_final.get("mAP_macro", 0.0),
                         results_final.get("inst_acc", 0.0))
                    )
                else:
                    logger.info(
                        ('Final eval | '
                        'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
                        'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f') %
                        (results_final.get("f1_micro", 0.0), results_final.get("f1_macro", 0.0),
                         results_final.get("p_micro", 0.0),  results_final.get("p_macro", 0.0),
                         results_final.get("r_micro", 0.0),  results_final.get("r_macro", 0.0),
                         results_final.get("roc_auc_macro", 0.0), results_final.get("mAP_macro", 0.0))
                    )

            # AOPCR with batch_size=1 for perturbation routine
            testloader_aopcr = DataLoader(
                testset, batch_size=1, shuffle=False,
                num_workers=args.num_workers, drop_last=False, pin_memory=True
            )
            print("Computing class-wise AOPCR with best model ...")
            aopcr_c, aopcr_w_avg, aopcr_mean, aopcr_sum, M_expl, M_rand, alphas, counts = compute_classwise_aopcr(
                eval_model,
                testloader_aopcr,
                args,
                stop=0.5,
                step=0.05,
                n_random=3,
                pred_threshold=0.5,
            )
            for c in range(args.num_classes):
                name = str(c)
                val = aopcr_c[c]
                cnt = int(counts[c])
                if np.isnan(val):
                    msg = f"[AOPCR] class {name} count={cnt} AOPCR=NaN"
                else:
                    msg = f"[AOPCR] class {name} count={cnt} AOPCR={val:.6f}"
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)
            if args.datatype == "original":
                summary_msg = f"Weighted AOPCR: {aopcr_w_avg:.6f}, Mean AOPCR: {aopcr_mean:.6f}, Sum AOPCR: {aopcr_sum:.6f}"
            else:
                summary_msg = f"Weighted AOPCR: {aopcr_w_avg:.6f}, Mean AOPCR: {aopcr_mean:.6f}"
            if logger is not None:
                logger.info(summary_msg)
            else:
                print(summary_msg)

    cleanup_distributed()


if __name__ == '__main__':
    main()