# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
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
import random

import wandb

# from timm.optim.adamp import AdamP
from lookhead import Lookahead
import warnings

from models.timemil import TimeMIL, newTimeMIL, AmbiguousMIL
from models.expmil import AmbiguousMILwithCL
from compute_aopcr import compute_classwise_aopcr

from os.path import join

# Suppress all warnings
warnings.filterwarnings("ignore")

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
    valid_class_mask = initialized.view(1,1,C)
    # valid_class_mask = (bag_label > 0).unsqueeze(1) & initialized.view(1, 1, C)  # [B,1,C]
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
def train(trainloader, milnet, criterion, optimizer, epoch, args, device, proto_bank=None, is_main=True, world_size=1):
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

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.to(device)
        bag_label = label.to(device)   # [B, C] multi-hot

        if args.dropout_patch > 0:
            selecy_window_indx = random.sample(range(10), int(args.dropout_patch * 10))
            inteval = int(len(bag_feats) // 10)
            for idx in selecy_window_indx:
                bag_feats[:, idx * inteval:idx * inteval + inteval, :] = torch.randn(1, device=device)

        optimizer.zero_grad()

        # forward
        if args.model == 'AmbiguousMIL':
            bag_prediction, instance_pred, weighted_instance_pred, non_weighted_instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(
                bag_feats
            )
        else:
            if epoch < args.epoch_des:
                bag_prediction = milnet(bag_feats, warmup=True)
                instance_pred = None
            else:
                bag_prediction = milnet(bag_feats, warmup=False)
                instance_pred = None

        # bag-level loss
        bag_loss = criterion(bag_prediction, bag_label)

        # 초기값 세팅
        inst_loss = 0.0
        ortho_loss = torch.tensor(0.0, device=device)
        smooth_loss = torch.tensor(0.0, device=device)
        sparsity_loss = torch.tensor(0.0, device=device)
        proto_inst_loss = torch.tensor(0.0, device=device)
        cls_contrast_loss = torch.tensor(0.0, device=device)

        if instance_pred is not None:
            # ---------- 기존 MIL 부분 ----------
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

            # ortho_loss = class_token_orthogonality_loss(x_cls)

            # pos_mask = (bag_label.sum(dim=0) > 0).float()   # [C]

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

            # diff = p_inst[:, 1:, :] - p_inst[:, :-1, :]   # [B, T-1, C]
            # diff = diff * pos_mask[None, None, :]
            # smooth_loss = (diff ** 2).mean()

            # ---------- prototype 기반 instance CL ----------
            if (epoch >= args.epoch_des) and (proto_bank is not None):
                proto_bank.update(x_cls.detach(), bag_label)
                if world_size > 1:
                    proto_bank.sync(world_size)
                proto_inst_loss = instance_prototype_contrastive_loss(
                    x_seq,
                    bag_label,
                    proto_bank,
                    tau=getattr(args, "proto_tau", 0.1),
                    sim_thresh=getattr(args, "proto_sim_thresh", 0.7),
                    win=getattr(args, "proto_win", 5),
                )

        # 최종 loss
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
        sum_proto_inst += proto_inst_loss.item()
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
                feats, label, y_inst = batch
            else:
                feats, label = batch
                y_inst = None
            
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
            # sparsity_loss = torch.tensor(0.0, device=device)
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
                if (epoch >= args.epoch_des) and (proto_bank is not None):
                    proto_inst_loss = instance_prototype_contrastive_loss(
                        x_seq,
                        bag_label,
                        proto_bank,
                        tau=getattr(args, "proto_tau", 0.1),
                        sim_thresh=getattr(args, "proto_sim_thresh", 0.7),
                        win=getattr(args, "proto_win", 5),
                    )

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
                        pred_inst = torch.argmax(attn_mean, dim=1).cpu()
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

    # ---------------- 데이터 구성 ----------------
    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:
        trainset = loadorean(args, split='train')
        testset = loadorean(args, split='test')

        seq_len,num_classes,L_in=trainset.max_len,trainset.num_class,trainset.feat_in

        if is_main:
            print(f'max length {seq_len}')
        args.feats_size = L_in
        args.num_classes =  num_classes
        if is_main:
            print(f'num class:{args.num_classes}' )

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

    # ---------------- 모델 구성 ----------------
    if args.model =='TimeMIL':
        base_model = TimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).to(device)
        proto_bank = None
    elif args.model == 'newTimeMIL':
        base_model = newTimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).to(device)
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
    if args.dataset in ['DuckDuckGeese','PenDigits','FingerMovements','BasicMotions','ERing','EigenWorms','HandMovementDirection','RacketSports','UWaveGestureLibrary']:
        batch = 64
    elif args.dataset in ['Heartbeat']:
        batch = 32
    elif args.dataset in ['EthanolConcentration','NATOPS','JapaneseVowels','MotorImagery','SelfRegulationSCP1']:
        batch = 16
    elif args.dataset in ['PEMS-SF','SelfRegulationSCP2','AtrialFibrillation','Cricket']:
        batch = 8
    elif args.dataset in ['StandWalkJump']:
        batch = 1
    elif args.dataset in ['Libras','Handwriting','Epilepsy','ArticularyWordRecognition','PhonemeSpectra']:
        batch = 128
    elif args.dataset in ['FaceDetection','LSST']:
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
        testset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True,
        sampler=test_sampler
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

        train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch, args, device, proto_bank=proto_bank, is_main=is_main, world_size=world_size)

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
            eval_model = newTimeMIL(args.feats_size, mDim=args.embed, n_classes=args.num_classes,
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