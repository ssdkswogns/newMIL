# -*- coding: utf-8 -*-
"""
DBA 전용 1D Conv 기반 Denoising AutoEncoder (datatype=original / mixed 모두 지원)

- original: build_dba_for_timemil(args) -> (trainset, testset, seq_len, num_classes, in_dim)
- mixed:    build_dba_windows_for_mixed(args) -> (Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, in_dim)
          이후 MixedSyntheticBagsConcatK로 bag을 구성
          (주의) mixed의 __getitem__ 반환이 (feats, label) 또는 (feats, label, y_inst)일 수 있으므로
          AE 학습에서는 feats만 사용하도록 안전 처리.

AE 학습은 unsupervised로 reconstruction만 수행합니다.
"""

import os
import argparse
import random
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dba_dataloader import build_dba_for_timemil, build_dba_windows_for_mixed
from syntheticdataset import MixedSyntheticBagsConcatK


# ------------------------
# Seed
# ------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_mkdir_p(p: str):
    os.makedirs(p, exist_ok=True)


# ------------------------
# Denoising utilities
# ------------------------
@torch.no_grad()
def apply_time_mask(x: torch.Tensor, mask_ratio: float):
    """
    x: [B,T,D]
    mask_ratio: 0~1
    """
    if mask_ratio <= 0:
        return x

    B, T, D = x.shape
    n_mask = max(1, int(T * mask_ratio))
    x_noisy = x.clone()
    for b in range(B):
        idx = torch.randperm(T, device=x.device)[:n_mask]
        x_noisy[b, idx, :] = 0.0
    return x_noisy


@torch.no_grad()
def apply_gaussian_noise(x: torch.Tensor, noise_std: float):
    if noise_std <= 0:
        return x
    return x + torch.randn_like(x) * noise_std


# ------------------------
# Model
# ------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, dropout=0.0):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        # self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Conv1DAE(nn.Module):
    """
    입력:  x [B,T,D]
    출력:  x_hat [B,T,D]
    임베딩: z [B,T,z_dim]
    """
    def __init__(
        self,
        in_dim: int,
        z_dim: int = 64,
        hidden: int = 128,
        depth: int = 4,
        k: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert depth >= 2

        enc = []
        ch = in_dim
        for _ in range(depth - 1):
            enc.append(ConvBlock(ch, hidden, k=k, s=1, dropout=dropout))
            ch = hidden
        self.encoder_backbone = nn.Sequential(*enc)
        self.to_z = nn.Conv1d(hidden, z_dim, kernel_size=1, bias=True)

        dec = []
        ch = z_dim
        for _ in range(depth - 1):
            dec.append(ConvBlock(ch, hidden, k=k, s=1, dropout=dropout))
            ch = hidden
        self.decoder_backbone = nn.Sequential(*dec)
        self.to_x = nn.Conv1d(hidden, in_dim, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.transpose(1, 2)        # [B,D,T]
        h = self.encoder_backbone(x_c) # [B,H,T]
        z_c = self.to_z(h)             # [B,z_dim,T]
        return z_c.transpose(1, 2)     # [B,T,z_dim]

    def forward(self, x: torch.Tensor):
        x_c = x.transpose(1, 2)
        h = self.encoder_backbone(x_c)
        z_c = self.to_z(h)

        d = self.decoder_backbone(z_c)
        xhat_c = self.to_x(d)
        x_hat = xhat_c.transpose(1, 2)
        z = z_c.transpose(1, 2)
        return x_hat, z


# ------------------------
# Batch unpack (mixed/original 공용)
# ------------------------
def unpack_feats(batch):
    """
    다양한 batch 구조를 안전하게 처리.
    - original DBA: (feats, label)
    - mixed DBA: (feats, label) 또는 (feats, label, y_inst)
    - 혹은 dataset 구현에 따라 list/tuple 길이가 달라질 수 있어도 feats는 0번으로 가정.
    """
    if isinstance(batch, (tuple, list)):
        feats = batch[0]
    else:
        feats = batch
    return feats


# ------------------------
# Train / Eval
# ------------------------
def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        feats = unpack_feats(batch)
        x = feats.to(device).float()  # [B,T,D]

        # denoising
        x_noisy = x
        x_noisy = apply_time_mask(x_noisy, args.time_mask_ratio)
        x_noisy = apply_gaussian_noise(x_noisy, args.noise_std)

        optimizer.zero_grad(set_to_none=True)
        x_hat, _ = model(x_noisy)

        loss = F.mse_loss(x_hat, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total += loss.item()
        n += 1

    return total / max(1, n)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        feats = unpack_feats(batch)
        x = feats.to(device).float()
        x_hat, _ = model(x)
        loss = F.mse_loss(x_hat, x)
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def dump_embeddings(model, loader, device, out_path: str, max_batches: int = -1):
    """
    loader의 순회 결과를 모아 Z=[N,T,z_dim]으로 저장.
    (DBA window가 크면 파일이 커질 수 있음)
    """
    model.eval()
    Z_list = []
    for bi, batch in enumerate(loader):
        feats = unpack_feats(batch)
        x = feats.to(device).float()
        z = model.encode(x).cpu().numpy()
        Z_list.append(z)
        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    Z = np.concatenate(Z_list, axis=0)
    maybe_mkdir_p(os.path.dirname(out_path))
    np.savez_compressed(out_path, Z=Z)
    print(f"[dump] saved embeddings: {out_path} | Z={Z.shape}")


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser("DBA Denoising AE (original/mixed)")

    # dataset flags (AmbiguousMIL 코드와 최대한 정합)
    parser.add_argument("--dataset", type=str, default="dba")
    parser.add_argument("--datatype", type=str, default="mixed", choices=["original", "mixed"])

    # DBA params
    parser.add_argument("--dba_root", type=str, default="./data/dba_data")
    parser.add_argument("--dba_window", type=int, default=12000)
    parser.add_argument("--dba_stride", type=int, default=6000)
    parser.add_argument("--dba_test_ratio", type=float, default=0.2)

    # mixed concat
    parser.add_argument("--concat_k", type=int, default=2, help="MixedSyntheticBagsConcatK concat_k")

    # AE arch
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)

    # denoising
    parser.add_argument("--time_mask_ratio", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.05)

    # train
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)

    # io
    parser.add_argument("--save_dir", type=str, default="./savemodel_ae/dba_ae")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--embed_out_name", type=str, default="dba_ae_embeddings.npz")
    parser.add_argument("--max_dump_batches", type=int, default=-1)

    args = parser.parse_args()
    assert args.dataset == "dba", "이 스크립트는 DBA 전용입니다."

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maybe_mkdir_p(args.save_dir)
    weights_dir = join(args.save_dir, "weights")
    maybe_mkdir_p(weights_dir)

    # --------------------
    # Build dataset
    # --------------------
    print(f"[DBA] datatype={args.datatype} | root={args.dba_root}")

    if args.datatype == "original":
        trainset, testset, seq_len, num_classes, in_dim = build_dba_for_timemil(args)
        print(f"[DBA-original] seq_len={seq_len}, in_dim={in_dim}, num_classes={num_classes}")

    else:  # mixed
        Xtr, ytr_idx, Xte, yte_idx, seq_len, num_classes, in_dim = build_dba_windows_for_mixed(args)

        trainset = MixedSyntheticBagsConcatK(
            X=Xtr,
            y_idx=ytr_idx,
            num_classes=num_classes,
            total_bags=len(Xtr),
            concat_k=args.concat_k,
            seed=args.seed,
            return_instance_labels=False,
        )
        testset = MixedSyntheticBagsConcatK(
            X=Xte,
            y_idx=yte_idx,
            num_classes=num_classes,
            total_bags=len(Xte),
            concat_k=args.concat_k,
            seed=args.seed + 1,
            return_instance_labels=True,   # mixed eval과 호환 (AE는 무시)
        )
        # mixed는 길이가 concat_k * window일 가능성이 큼
        print(f"[DBA-mixed] base_window={seq_len} (from builder), concat_k={args.concat_k} "
              f"| in_dim={in_dim}, num_classes={num_classes}")

    trainloader = DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )

    # --------------------
    # Model / Optim
    # --------------------
    model = Conv1DAE(
        in_dim=in_dim,
        z_dim=args.z_dim,
        hidden=args.hidden,
        depth=args.depth,
        k=args.kernel,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = join(weights_dir, "best_ae.pth")

    # save options
    opt_path = join(args.save_dir, "option.txt")
    with open(opt_path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    # --------------------
    # Train loop
    # --------------------
    print(f"[AE] start training | device={device} | save_dir={args.save_dir}")
    for epoch in range(1, args.num_epochs + 1):
        tr = train_one_epoch(model, trainloader, optimizer, device, args)
        va = eval_one_epoch(model, testloader, device)

        print(f"Epoch {epoch:04d}/{args.num_epochs} | train_mse={tr:.6f} | val_mse={va:.6f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), best_path)
            print(f"  [best] saved: {best_path} (val_mse={best_val:.6f})")

    # --------------------
    # Dump embeddings (optional)
    # --------------------
    if args.save_embeddings:
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        out_path = join(args.save_dir, args.embed_out_name)

        # train/test를 구분 저장하고 싶으면 파일명을 나누십시오.
        full_loader = DataLoader(
            torch.utils.data.ConcatDataset([trainset, testset]),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        dump_embeddings(model, full_loader, device, out_path, max_batches=args.max_dump_batches)

    print("[AE] done.")


if __name__ == "__main__":
    main()
