# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
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

from models.timemil_old import TimeMIL, newTimeMIL, AmbiguousMIL

# Suppress all warnings
warnings.filterwarnings("ignore")

seed = 42

random.seed(seed)             # python random
np.random.seed(seed)          # numpy random
torch.manual_seed(seed)       # CPU
torch.cuda.manual_seed(seed)  # GPU 단일
torch.cuda.manual_seed_all(seed)  # multi-GPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def temporal_similarity_smooth_loss(x_seq):
    """
    x_seq : (B, L, D)  # 시퀀스 토큰 임베딩 (어텐션 레이어 출력)
    return: scalar TV-L1 loss on similarity matrix
    """
    B, L, D = x_seq.shape
    # 1) cosine 유사도 위해 정규화
    x_norm = F.normalize(x_seq, dim=-1)                 # (B, L, D)
    S = torch.matmul(x_norm, x_norm.transpose(1, 2))    # (B, L, L), cos-sim

    # 2) TV-L1: ∥∇x S∥_1 + ∥∇y S∥_1
    #   가로(열) 방향: S[:, :, 1:] - S[:, :, :-1]
    grad_x = S[:, :, 1:] - S[:, :, :-1]                 # (B, L, L-1)
    #   세로(행) 방향: S[:, 1:, :] - S[:, :-1, :]
    grad_y = S[:, 1:, :] - S[:, :-1, :]                 # (B, L-1, L)

    # 3) L1 누적 (배치/길이 축 정규화)
    loss = grad_x.abs().mean() + grad_y.abs().mean()
    return loss

def class_token_orthogonality_loss(x_cls, diag_weight=1.0, offdiag_weight=1.0, eps=1e-6):
    B, C, D = x_cls.shape
    x_norm = F.normalize(x_cls, dim=-1)
    S = torch.matmul(x_norm, x_norm.transpose(1, 2))          # (B, C, C)

    I = torch.eye(C, device=x_cls.device).unsqueeze(0).expand(B, C, C)

    diag = S.diagonal(dim1=1, dim2=2)                         # (B, C)
    diag_loss = ((diag - 1.0) ** 2).mean() if diag_weight > 0 else 0.0

    off_mask = ~torch.eye(C, dtype=torch.bool, device=x_cls.device).unsqueeze(0).expand(B, C, C)
    off = (S - I).masked_select(off_mask)
    offdiag_loss = (off ** 2).mean() if offdiag_weight > 0 else 0.0

    return diag_weight * diag_loss + offdiag_weight * offdiag_loss

# def train(trainloader, milnet, criterion, optimizer, epoch, args):
#     milnet.train()
#     total_loss = 0
#     sum_bag = 0.0
#     sum_inst = 0.0
#     sum_ortho = 0.0
#     sum_smooth = 0.0
#     sum_total = 0.0
#     n = 0
#     for batch_id, (feats, label) in enumerate(trainloader):
#         bag_feats = feats.cuda()
#         bag_label = label.cuda()
#         if args.dropout_patch>0:
#             selecy_window_indx = random.sample(range(10),int(args.dropout_patch*10))
#             inteval = int(len(bag_feats)//10)
#             for idx in selecy_window_indx:
#                 bag_feats[:,idx*inteval:idx*inteval+inteval,:] = torch.randn(1).cuda()
#         optimizer.zero_grad()
#         if args.model == 'AmbiguousMIL':
#             if epoch<args.epoch_des:
#                 bag_prediction, instance_pred, x_cls, x_seq, attn_layer1, attn_layer2  = milnet(bag_feats,warmup = True)
#             else:
#                 bag_prediction, instance_pred, x_cls, x_seq, attn_layer1, attn_layer2  = milnet(bag_feats,warmup = False)
#         else:
#             if epoch<args.epoch_des:
#                 bag_prediction  = milnet(bag_feats,warmup = True)
#                 instance_pred = None
#             else:
#                 bag_prediction  = milnet(bag_feats,warmup = False)
#                 instance_pred = None
#         bag_loss = criterion(bag_prediction, bag_label)
#         inst_loss = 0.0
#         ortho_loss = torch.tensor(0.0, device=bag_feats.device)
#         smooth_loss = torch.tensor(0.0, device=bag_feats.device)
#         if instance_pred is not None:
#             p_inst = torch.sigmoid(instance_pred)
#             eps = 1e-6
#             p_bag_from_inst = 1 - torch.prod(torch.clamp(1 - p_inst, min=eps), dim=1)
#             inst_loss = F.binary_cross_entropy(p_bag_from_inst, bag_label.float())
#             ortho_loss = class_token_orthogonality_loss(x_cls)
#             smooth_loss = temporal_similarity_smooth_loss(x_seq)
#             sparsity_per_class = p_inst.mean(dim=(0,1))
#         if args.model == 'AmbiguousMIL':
#             loss = args.bag_loss_w*bag_loss + args.inst_loss_w*inst_loss + args.ortho_loss_w*ortho_loss + args.smooth_loss_w*smooth_loss
#             sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % \
#                 (batch_id, len(trainloader), bag_loss.item(),loss.item()))
#         else:
#             loss = bag_loss 
#             sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % \
#                             (batch_id, len(trainloader), bag_loss.item(),loss.item()))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
#         optimizer.step()
#         total_loss += bag_loss.item()
#         sum_bag += bag_loss.item()
#         sum_inst += float(inst_loss) if isinstance(inst_loss, float) else inst_loss.item()
#         sum_ortho += ortho_loss.item()
#         sum_smooth += smooth_loss.item()
#         sum_total += loss.item()
#         n += 1
#     wandb.log({
#         "epoch": epoch,
#         "train/bag_loss": sum_bag / max(1, n),
#         "train/inst_loss": sum_inst / max(1, n),
#         "train/ortho_loss": sum_ortho / max(1, n),
#         "train/smooth_loss": sum_smooth / max(1, n),
#         "train/total_loss": sum_total / max(1, n),
#     }, step=epoch)
#     return total_loss / len(trainloader)

def train(trainloader, milnet, criterion, optimizer, epoch, args):
    milnet.train()
    total_loss = 0
    sum_bag = 0.0
    sum_inst = 0.0
    sum_ortho = 0.0
    sum_smooth = 0.0
    sum_sparsity = 0.0
    sum_total = 0.0
    n = 0

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.cuda()
        bag_label = label.cuda()   # [B, C] multi-hot

        if args.dropout_patch > 0:
            selecy_window_indx = random.sample(range(10), int(args.dropout_patch * 10))
            inteval = int(len(bag_feats) // 10)
            for idx in selecy_window_indx:
                bag_feats[:, idx * inteval:idx * inteval + inteval, :] = torch.randn(1).cuda()

        optimizer.zero_grad()

        # forward
        if args.model == 'AmbiguousMIL':
            if epoch < args.epoch_des:
                bag_prediction, instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(
                    bag_feats, warmup=True
                )
            else:
                bag_prediction, instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(
                    bag_feats, warmup=False
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
        ortho_loss = torch.tensor(0.0, device=bag_feats.device)
        smooth_loss = torch.tensor(0.0, device=bag_feats.device)
        sparsity_loss = torch.tensor(0.0, device=bag_feats.device)   # [NEW]

        if instance_pred is not None:
            # instance → bag MIL loss
            p_inst = torch.sigmoid(instance_pred)        # [B, T, C]
            # p_inst = torch.softmax(instance_pred, dim=2)
            eps = 1e-6
            p_bag_from_inst = 1 - torch.prod(
                torch.clamp(1 - p_inst, min=eps), dim=1
            )                                           # [B, C]
            inst_loss = F.binary_cross_entropy(
                p_bag_from_inst, bag_label.float()
            )

            # class token orthogonality
            ortho_loss = class_token_orthogonality_loss(x_cls)

            # [NEW] 이번 배치에서 실제로 등장한 positive class만 선택
            pos_mask = (bag_label.sum(dim=0) > 0).float()   # [C]

            # [NEW] sparsity loss (positive class만)
            sparsity_per_class = p_inst.mean(dim=(0, 1))    # [C]
            if pos_mask.sum() > 0:
                sparsity_loss = (sparsity_per_class * pos_mask).sum() / (
                    pos_mask.sum() + 1e-6
                )
            else:
                sparsity_loss = torch.tensor(0.0, device=bag_feats.device)
            sparsity_loss = sparsity_per_class.mean()  # pos_mask 제거된 기본 버전

            # [NEW] temporal smoothness loss: (p_t - p_{t-1})^2
            diff = p_inst[:, 1:, :] - p_inst[:, :-1, :]   # [B, T-1, C]

            # positive class에만 smoothness 적용
            diff = diff * pos_mask[None, None, :]          # broadcasting

            smooth_loss = (diff ** 2).mean()

        # 최종 loss
        if args.model == 'AmbiguousMIL':
            # [NEW] sparsity_loss를 weight와 함께 추가 (args.sparsity_loss_w 필요)
            loss = (
                args.bag_loss_w * bag_loss
                + args.inst_loss_w * inst_loss
                + args.ortho_loss_w * ortho_loss
                + args.smooth_loss_w * smooth_loss
                + args.sparsity_loss_w * sparsity_loss    # [NEW]
            )
            sys.stdout.write(
                '\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' %
                (batch_id, len(trainloader), bag_loss.item(), loss.item())
            )
        else:
            loss = bag_loss
            sys.stdout.write(
                '\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' %
                (batch_id, len(trainloader), bag_loss.item(), loss.item())
            )

        # backward & step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()

        # 통계값 누적
        total_loss += bag_loss.item()
        sum_bag += bag_loss.item()
        sum_inst += float(inst_loss) if isinstance(inst_loss, float) else inst_loss.item()
        sum_ortho += ortho_loss.item()
        sum_smooth += smooth_loss.item()
        sum_sparsity += sparsity_loss.item()    # [NEW]
        sum_total += loss.item()
        n += 1

    # wandb logging
    wandb.log({
        "epoch": epoch,
        "train/bag_loss": sum_bag / max(1, n),
        "train/inst_loss": sum_inst / max(1, n),
        "train/ortho_loss": sum_ortho / max(1, n),
        "train/smooth_loss": sum_smooth / max(1, n),
        "train/sparsity_loss": sum_sparsity / max(1, n),  # [NEW]
        "train/total_loss": sum_total / max(1, n),
    }, step=epoch)

    return total_loss / len(trainloader)

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def test(testloader, milnet, criterion, epoch, args, threshold: float = 0.5):
    milnet.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    sum_bag = 0.0
    sum_inst = 0.0
    sum_ortho = 0.0
    sum_smooth = 0.0
    sum_sparsity = 0.0   # [NEW]
    sum_total = 0.0
    n = 0

    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.cuda()
            bag_label = label.cuda()   # [B, C]

            if args.model == 'AmbiguousMIL':
                bag_prediction, instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(bag_feats)
            else:
                bag_prediction = milnet(bag_feats)
                instance_pred = None

            # bag-level loss
            bag_loss = criterion(bag_prediction, bag_label)

            inst_loss = 0.0
            ortho_loss = torch.tensor(0.0, device=bag_feats.device)
            smooth_loss = torch.tensor(0.0, device=bag_feats.device)
            sparsity_loss = torch.tensor(0.0, device=bag_feats.device)  # [NEW]

            if instance_pred is not None:
                # instance → bag MIL loss
                p_inst = torch.sigmoid(instance_pred)        # [B, T, C]
                # p_inst = torch.softmax(instance_pred, dim=2)
                eps = 1e-6
                p_bag_from_inst = 1 - torch.prod(
                    torch.clamp(1 - p_inst, min=eps), dim=1
                )                                           # [B, C]
                inst_loss = F.binary_cross_entropy(
                    p_bag_from_inst, bag_label.float()
                )

                # class token orthogonality
                ortho_loss = class_token_orthogonality_loss(x_cls)

                # [NEW] 이번 배치에서 실제로 등장한 positive class만 선택
                pos_mask = (bag_label.sum(dim=0) > 0).float()   # [C]

                # [NEW] sparsity loss (positive class만)
                sparsity_per_class = p_inst.mean(dim=(0, 1))    # [C]
                if pos_mask.sum() > 0:
                    sparsity_loss = (sparsity_per_class * pos_mask).sum() / (
                        pos_mask.sum() + 1e-6
                    )
                else:
                    sparsity_loss = torch.tensor(0.0, device=bag_feats.device)
                sparsity_loss = sparsity_per_class.mean()   # pos_mask 제거 버전

                # [NEW] temporal smoothness loss: (p_t - p_{t-1})^2, positive class만
                diff = p_inst[:, 1:, :] - p_inst[:, :-1, :]   # [B, T-1, C]

                diff = diff * pos_mask[None, None, :]         # broadcast

                smooth_loss = (diff ** 2).mean()

            # total loss
            if args.model == 'AmbiguousMIL':
                loss = (
                    args.bag_loss_w * bag_loss
                    + args.inst_loss_w * inst_loss
                    + args.ortho_loss_w * ortho_loss
                    + args.smooth_loss_w * smooth_loss
                    + args.sparsity_loss_w * sparsity_loss   # [NEW]
                )
                sys.stdout.write(
                    '\r Testing bag [%d/%d] bag loss: %.4f  total loss: %.4f' %
                    (batch_id, len(testloader), bag_loss.item(), loss.item())
                )
            else:
                loss = bag_loss
                sys.stdout.write(
                    '\r Testing bag [%d/%d] bag loss: %.4f  total loss: %.4f' %
                    (batch_id, len(testloader), bag_loss.item(), loss.item())
                )

            total_loss += loss.item()

            probs = torch.sigmoid(bag_prediction).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label.cpu().numpy())

            sum_bag += bag_loss.item()
            sum_inst += float(inst_loss) if isinstance(inst_loss, float) else inst_loss.item()
            sum_ortho += ortho_loss.item()
            sum_smooth += smooth_loss.item()
            sum_sparsity += sparsity_loss.item()   # [NEW]
            sum_total += loss.item()
            n += 1

    # metric 계산
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= threshold).astype(np.int32)

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

    # wandb logging
    wandb.log({
        "val/bag_loss": sum_bag / max(1, n),
        "val/inst_loss": sum_inst / max(1, n),
        "val/ortho_loss": sum_ortho / max(1, n),
        "val/smooth_loss": sum_smooth / max(1, n),
        "val/sparsity_loss": sum_sparsity / max(1, n),  # [NEW]
        "val/total_loss": sum_total / max(1, n),
    }, step=epoch)

    results = {
        "f1_micro": f1_micro, "f1_macro": f1_macro,
        "p_micro": p_micro,   "p_macro": p_macro,
        "r_micro": r_micro,   "r_macro": r_macro,
        "roc_auc_macro": roc_macro, "mAP_macro": ap_macro
    }
    return total_loss / len(testloader), results

# def test(testloader, milnet, criterion, epoch, args, threshold: float = 0.5):
#     milnet.eval()
#     total_loss = 0.0
#     all_labels = []
#     all_probs  = []
#     sum_bag = 0.0
#     sum_inst = 0.0
#     sum_ortho = 0.0
#     sum_smooth = 0.0
#     sum_total = 0.0
#     n = 0
#     with torch.no_grad():
#         for batch_id, (feats, label) in enumerate(testloader):
#             bag_feats = feats.cuda()
#             bag_label = label.cuda()
#             if args.model == 'AmbiguousMIL':
#                 bag_prediction, instance_pred, x_cls, x_seq, attn_layer1, attn_layer2 = milnet(bag_feats)
#             else:
#                 bag_prediction = milnet(bag_feats)
#                 instance_pred = None
#             bag_loss = criterion(bag_prediction, bag_label)
#             inst_loss = 0.0
#             ortho_loss = torch.tensor(0.0, device=bag_feats.device)
#             smooth_loss = torch.tensor(0.0, device=bag_feats.device)
#             if instance_pred is not None:
#                 p_inst = torch.sigmoid(instance_pred)
#                 eps = 1e-6
#                 p_bag_from_inst = 1 - torch.prod(torch.clamp(1 - p_inst, min=eps), dim=1)
#                 inst_loss = F.binary_cross_entropy(p_bag_from_inst, bag_label.float())
#                 ortho_loss = class_token_orthogonality_loss(x_cls)
#                 smooth_loss = temporal_similarity_smooth_loss(x_seq)
#             if args.model == 'AmbiguousMIL':
#                 loss = args.bag_loss_w*bag_loss + args.inst_loss_w*inst_loss + args.ortho_loss_w*ortho_loss + args.smooth_loss_w*smooth_loss
#                 sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f  total loss: %.4f' %
#                                  (batch_id, len(testloader), bag_loss.item(), loss.item()))
#             else:
#                 loss = bag_loss
#                 sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f  total loss: %.4f' %
#                                  (batch_id, len(testloader), bag_loss.item(), loss.item()))
#             total_loss += loss.item()
#             probs = torch.sigmoid(bag_prediction).cpu().numpy()
#             all_probs.append(probs)
#             all_labels.append(label.cpu().numpy())
#             sum_bag += bag_loss.item()
#             sum_inst += float(inst_loss) if isinstance(inst_loss, float) else inst_loss.item()
#             sum_ortho += ortho_loss.item()
#             sum_smooth += smooth_loss.item()
#             sum_total += loss.item()
#             n += 1
#     y_true = np.vstack(all_labels)
#     y_prob = np.vstack(all_probs)
#     y_pred = (y_prob >= threshold).astype(np.int32)
#     f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
#     f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
#     p_micro  = precision_score(y_true, y_pred, average='micro', zero_division=0)
#     p_macro  = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     r_micro  = recall_score(y_true, y_pred, average='micro', zero_division=0)
#     r_macro  = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     roc_list, ap_list = [], []
#     for c in range(y_true.shape[1]):
#         if len(np.unique(y_true[:, c])) == 2:
#             try:
#                 roc_list.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
#                 ap_list.append(average_precision_score(y_true[:, c], y_prob[:, c]))
#             except Exception:
#                 pass
#     roc_macro = float(np.mean(roc_list)) if roc_list else 0.0
#     ap_macro  = float(np.mean(ap_list))  if ap_list  else 0.0
#     wandb.log({
#         "val/bag_loss": sum_bag / max(1, n),
#         "val/inst_loss": sum_inst / max(1, n),
#         "val/ortho_loss": sum_ortho / max(1, n),
#         "val/smooth_loss": sum_smooth / max(1, n),
#         "val/total_loss": sum_total / max(1, n),
#     }, step=epoch)
#     results = {
#         "f1_micro": f1_micro, "f1_macro": f1_macro,
#         "p_micro": p_micro,   "p_macro": p_macro,
#         "r_micro": r_micro,   "r_macro": r_macro,
#         "roc_auc_macro": roc_macro, "mAP_macro": ap_macro
#     }
#     return total_loss / len(testloader), results


def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="PenDigits", type=str, help='dataset ')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512] resnet-50 1024')
    parser.add_argument('--lr', default=5e-3, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.5, type=float, help='Patch dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
    parser.add_argument('--model', default='TimeMIL', type=str, help='MIL model')
    parser.add_argument('--prepared_npz', type=str, default='./data/PAMAP2.npz', help='전처리 결과 npz 경로 (예: ./data/PAMAP2.npz)')
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')
    
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=10, type=int, help='turn on warmup')
   
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')               
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    args.save_dir = args.save_dir+'InceptBackbone'
    maybe_mkdir_p(join(args.save_dir, f'{args.dataset}'))
    exp_path = make_dirs(join(args.save_dir, f'{args.dataset}'))
    args.save_dir = exp_path
    maybe_mkdir_p(args.save_dir)

    version_name = os.path.basename(exp_path)

    wandb.init(project="TimeMIL", name=f"{args.dataset}_{args.model}_{version_name}", config=vars(args))
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*",   step_metric="epoch")
    wandb.define_metric("score/*", step_metric="epoch")

    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)

    args.bag_loss_w = 0.7
    args.inst_loss_w = 0.2
    args.ortho_loss_w = 0.1
    args.smooth_loss_w = 0.02
    args.sparsity_loss_w = 0.02

    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    criterion = nn.BCEWithLogitsLoss()
    
    ###################################
    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:
       
        trainset = loadorean(args, split='train')
        testset = loadorean(args, split='test')
        
        seq_len,num_classes,L_in=trainset.max_len,trainset.num_class,trainset.feat_in      
        
        print(f'max lenght {seq_len}')
        args.feats_size = L_in
        args.num_classes =  num_classes
        print(f'num class:{args.num_classes}' )

    elif args.dataset in ['PAMAP2']:
        trainset = loadorean(args, split='train')
        testset  = loadorean(args, split='test')
        seq_len, num_classes, L_in = trainset.max_len, trainset.num_class, trainset.feat_in

        print(f'max lenght {seq_len}')
        args.feats_size  = L_in
        args.num_classes = num_classes
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

        # trainset = MixedSyntheticBags(
        #     X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
        #     total_bags=max(2000, len(Xtr)),
        #     probs=mix_probs, total_len=SYN_TOTAL_LEN,
        #     min_seg_len=32, ensure_distinct_classes=True, seed=args.seed
        # )

        # testset = MixedSyntheticBags(
        #     X=Xte, y_idx=yte_idx, num_classes=num_classes,
        #     total_bags=max(1000, len(Xte)),
        #     probs=mix_probs, total_len=SYN_TOTAL_LEN,
        #     min_seg_len=32, ensure_distinct_classes=True, seed=args.seed+1
        # )

        trainset = MixedSyntheticBagsConcatK(
            X=Xtr, y_idx=ytr_idx, num_classes=num_classes,
            total_bags=len(Xtr),
            seed=args.seed
        )

        testset = MixedSyntheticBagsConcatK(
            X=Xte, y_idx=yte_idx, num_classes=num_classes,
            total_bags=len(Xte),
            seed=args.seed+1
        )

        args.feats_size = L_in
        print(f'num class: {args.num_classes}')
        print(f'total_len: {SYN_TOTAL_LEN}')
    
    if args.model =='TimeMIL':
        milnet = TimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).cuda()
    elif args.model == 'newTimeMIL':
        milnet = newTimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).cuda()
    elif args.model == 'AmbiguousMIL':
        milnet = AmbiguousMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len, is_instance=True).cuda()
    else:
        raise Exception("Model not available")
    
    if  args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer, alpha=0.5, k=5)    
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer, alpha=0.5, k=5) 
    
    if args.dataset in ['ArticularyWordRecognition','BasicMotions','Cricket','ERing','EigenWorms','HandMovementDirection','LSST','PhonemeSpectra','RacketSports','UWaveGestureLibrary']:
        batch = 8
    elif args.dataset in ['FingerMovements','Handwriting','Heartbeat']:
        batch = 4
    elif args.dataset in ['DuckDuckGeese','EthanolConcentration','NATOPS','JapaneseVowels','MotorImagery','SelfRegulationSCP1','SelfRegulationSCP2']:
        batch = 2
    elif args.dataset in ['Epilepsy','FaceDetection','Libras','PEMS-SF', 'StandWalkJump']:
        batch = 1
    elif args.dataset in ['StandWalkJump','AtrialFibrillation']:
        batch = 1
    elif args.dataset in ['PenDigits']:
        batch = 16
    else:
        batch = args.batchsize

    trainloader = DataLoader(trainset, batch, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    testloader = DataLoader(testset, batch, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    
    os.makedirs(join(args.save_dir,'lesion'), exist_ok=True)
    results_best = None
    for epoch in range(1, args.num_epochs + 1):

        train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch,args)      
        test_loss_bag, results = test(testloader, milnet, criterion, epoch, args, threshold=0.5)
        wandb.log({
            "epoch": epoch,
            "score/f1_micro": results["f1_micro"],
            "score/f1_macro": results["f1_macro"],
            "score/precision_micro": results["p_micro"],
            "score/precision_macro": results["p_macro"],
            "score/recall_micro": results["r_micro"],
            "score/recall_macro": results["r_macro"],
            "score/roc_auc_macro": results["roc_auc_macro"],
            "score/mAP_macro": results["mAP_macro"]
        }, step=epoch)
        
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

        current_score = results["mAP_macro"]

        if current_score >= best_score:
            
            results_best = results
            
            best_score = current_score
            print(current_score)
            save_name = os.path.join(save_path, f'best_{args.model}.pth')

            torch.save(milnet.state_dict(), save_name)
            logger.info('Best model saved at: ' + save_name)
    
    best = results_best
    logger.info(
        ('Best Results | '
        'F1(mi)=%.4f F1(Ma)=%.4f  P(mi)=%.4f P(Ma)=%.4f  R(mi)=%.4f R(Ma)=%.4f  '
        'ROC_AUC(Ma)=%.4f  mAP(Ma)=%.4f') %
        (best["f1_micro"], best["f1_macro"],
        best["p_micro"],  best["p_macro"],
        best["r_micro"],  best["r_macro"],
        best["roc_auc_macro"], best["mAP_macro"])
    )


        # if args.weight_div>0:
        #     if epoch%10==0:
        #         print('--------------------Clustering--------------------\n')
        #         cluster_idx_dict = pre_cluter(trainloader, milnet, criterion, optimizer, args,init= False)
        #         print('--------------------Clustering finished--------------------\n')

if __name__ == '__main__':
    main()
