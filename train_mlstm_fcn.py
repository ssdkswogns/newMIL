#!/usr/bin/env python3
"""
Minimal training script for the PyTorch MLSTM_FCN model on aeon classification datasets.
Assumes single-label classification.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from aeon.datasets import load_classification
from models.MLSTM_FCN import MLSTM_FCN


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_dataset(name: str, extract_path: str):
    res_tr = load_classification(name=name, split="train", extract_path=extract_path, return_metadata=True)
    res_te = load_classification(name=name, split="test", extract_path=extract_path, return_metadata=True)

    if len(res_tr) == 3:
        Xtr, ytr, meta = res_tr
    else:
        Xtr, ytr = res_tr
        meta = {"class_values": np.unique(ytr)}

    if len(res_te) == 3:
        Xte, yte, _ = res_te
    else:
        Xte, yte = res_te

    classes = list(meta["class_values"])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    ytr_idx = np.array([cls_to_idx[y] for y in ytr], dtype=np.int64)
    yte_idx = np.array([cls_to_idx[y] for y in yte], dtype=np.int64)

    return (Xtr, ytr_idx), (Xte, yte_idx), len(classes)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y = []
    all_p = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_p.append(preds)
        all_y.append(yb.numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, bal, f1m


def main():
    parser = argparse.ArgumentParser(description="Train MLSTM_FCN on aeon dataset")
    parser.add_argument("--dataset", type=str, required=True, help="aeon dataset name")
    parser.add_argument("--extract-path", type=str, default="./data", help="Path for aeon datasets")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batchsize", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    set_seed(args.seed)

    (Xtr, ytr), (Xte, yte), num_classes = load_dataset(args.dataset, args.extract_path)
    # X: (N, T, C) -> torch [N, T, C]
    Xtr_t = torch.from_numpy(Xtr).float()
    Xte_t = torch.from_numpy(Xte).float()
    ytr_t = torch.from_numpy(ytr).long()
    yte_t = torch.from_numpy(yte).long()

    train_ds = TensorDataset(Xtr_t, ytr_t)
    test_ds = TensorDataset(Xte_t, yte_t)
    train_loader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batchsize, shuffle=False)

    input_dim = Xtr.shape[-1]
    model = MLSTM_FCN(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc, bal, f1m = evaluate(model, test_loader, device)
        if f1m > best_f1:
            best_f1 = f1m
        print(f"[{epoch:03d}] loss={loss:.4f} acc={acc:.4f} bal={bal:.4f} f1m={f1m:.4f} (best f1m={best_f1:.4f})")


if __name__ == "__main__":
    main()
