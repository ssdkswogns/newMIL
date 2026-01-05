"""
PyTorch implementation of the MLSTM-FCN architecture:
- LSTM branch on the raw sequence.
- Temporal Conv1d branch with squeeze-and-excitation.
- Concatenate both branches and classify.

Expected input: x shaped [B, T, C] (time-major). Outputs logits [B, num_classes].
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        b, c, _ = x.shape
        w = F.adaptive_avg_pool1d(x, 1).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class MLSTM_FCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lstm_hidden: int = 8,
        conv_filters: Tuple[int, int, int] = (128, 256, 128),
        kernel_sizes: Tuple[int, int, int] = (8, 5, 3),
        lstm_dropout: float = 0.8,
        se_reduction: int = 16,
    ):
        super().__init__()
        f1, f2, f3 = conv_filters
        k1, k2, k3 = kernel_sizes

        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        # Conv branch
        self.conv1 = nn.Conv1d(input_dim, f1, kernel_size=k1, padding=k1 // 2)
        self.bn1 = nn.BatchNorm1d(f1)
        self.se1 = SqueezeExcite(f1, se_reduction)

        self.conv2 = nn.Conv1d(f1, f2, kernel_size=k2, padding=k2 // 2)
        self.bn2 = nn.BatchNorm1d(f2)
        self.se2 = SqueezeExcite(f2, se_reduction)

        self.conv3 = nn.Conv1d(f2, f3, kernel_size=k3, padding=k3 // 2)
        self.bn3 = nn.BatchNorm1d(f3)

        # Classifier
        self.classifier = nn.Linear(lstm_hidden + f3, num_classes)

    def forward(self, x: torch.Tensor, return_cam: bool = False):
        """
        x: [B, T, C]
        returns logits [B, num_classes] (or (logits, cam) if return_cam)
        """
        # LSTM branch
        lstm_out, (h_n, _) = self.lstm(x)  # h_n: [1, B, H]
        lstm_feat = self.lstm_dropout(h_n.squeeze(0))  # [B, H]

        # Conv branch (use channels-first)
        y = x.transpose(1, 2)  # [B, C, T]
        y = F.relu(self.bn1(self.conv1(y)))
        y = self.se1(y)

        y = F.relu(self.bn2(self.conv2(y)))
        y = self.se2(y)

        y = F.relu(self.bn3(self.conv3(y)))  # y: [B, f3, T]
        y_map = y  # for CAM
        y = F.adaptive_avg_pool1d(y, 1).squeeze(-1)  # [B, f3]

        feat = torch.cat([lstm_feat, y], dim=1)
        logits = self.classifier(feat)
        if not return_cam:
            return logits

        # Class Activation Map using classifier weights for conv branch
        w = self.classifier.weight[:, lstm_feat.shape[1]:]  # [num_classes, f3]
        cam = torch.einsum("cf,bft->bct", w, y_map)         # [B, C, T]
        return logits, cam


if __name__ == "__main__":
    # Small sanity check
    model = MLSTM_FCN(input_dim=16, num_classes=5)
    dummy = torch.randn(2, 50, 16)
    out = model(dummy)
    print(out.shape)  # [2, 5]
