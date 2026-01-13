"""
HybridMIL and ContextualMIL models.

HybridMIL: Dual pooling strategy (Attention + Conjunctive pooling)
ContextualMIL: HybridMIL + Self-attention context injection
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inceptiontime import InceptionTimeFeatureExtractor


class SinusoidalPositionalEncoding(nn.Module):
    """Standard 1D sinusoidal positional encoding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        device = x.device
        pos = torch.arange(T, device=device, dtype=x.dtype).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=x.dtype) * (-math.log(10000.0) / D)
        )
        pe = torch.zeros(T, D, device=device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        return x + pe


class _SelfAttnBlock(nn.Module):
    """Lightweight Transformer-style block for sequence context."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class HybridMIL(nn.Module):
    """
    HybridMIL (AmbiguousMILwithCL from expmil.py)

    InceptionTime backbone + Dual pooling strategy:
      1) Attention pooling for bag prediction (TS)
      2) Conjunctive pooling for time-point prediction (TP)
    + Contrastive projection between bag embedding & time-point embedding.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        mDim: int = 128,
        dropout: float = 0.0,
        is_instance: bool = True,
    ) -> None:
        super().__init__()
        self.is_instance = is_instance
        self.n_classes = n_classes
        self.mDim = mDim

        # ------------------------------
        # 1) Backbone (InceptionTime)
        # ------------------------------
        out_channels = max(1, mDim // 4)
        self.feature_extractor = InceptionTimeFeatureExtractor(
            n_in_channels=in_features
        )

        # ------------------------------
        # 2) Attention pooling head (TS prediction)
        # ------------------------------
        self.attn_head_ts = nn.Sequential(
            nn.Linear(mDim, 8*n_classes),
            nn.Tanh(),
            nn.Linear(8*n_classes, n_classes),
            nn.Sigmoid(),
        )
        self.bag_classifier_ts = nn.Linear(mDim, 1)

        # ------------------------------
        # 3) Conjunctive pooling head (TP prediction)
        # ------------------------------
        self.attn_head_tp = nn.Sequential(
            nn.Linear(mDim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.inst_classifier = nn.Linear(mDim, n_classes)

        self.dropout_p = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, warmup: bool = False):
        """
        Args:
            x: [B, T, D_in]
            warmup: bool (for compatibility with training loop)

        Returns:
            bag_logits_ts:   [B, C]  (attention pooling 기반 TS prediction)
            bag_logits_tp:   [B, C]  (conjunctive pooling 기반 bag prediction)
            weighted_logits: [B, T, C] (weighted instance logits)
            inst_logits:     [B, T, C] (time point prediction)
            bag_emb:         [B, C, mDim] (bag embeddings for CL)
            inst_emb:        [B, T, mDim] (instance embeddings for CL)
            attn_ts:         [B, C, T] (attention pooling weights)
            attn_tp:         [B, T, 1] (conjunctive pooling weights)
        """
        B, T, D_in = x.shape

        # 1) Backbone
        x = x.transpose(1, 2)             # [B, D_in, T]
        feats = self.feature_extractor(x) # [B, mDim, T]

        if self.dropout_p > 0:
            feats = self.dropout(feats)

        inst_emb = feats.transpose(1, 2)  # [B, T, mDim]

        # ------------------------------
        # 2) Attention pooling (TS)
        # ------------------------------
        score_ts = self.attn_head_ts(inst_emb)              # [B, T, C]
        score_ts = score_ts.permute(0, 2, 1)               # [B, C, T]
        attn_ts = F.softmax(score_ts, dim=-1)              # [B, C, T]

        bag_emb = torch.bmm(attn_ts, inst_emb)        # [B, C, mDim]
        B, C, D = bag_emb.shape
        bag_emb_flat = bag_emb.reshape(B * C, D)      # [B*C, mDim]
        bag_logits_flat = self.bag_classifier_ts(bag_emb_flat)  # [B*C, 1]
        bag_logits_ts = bag_logits_flat.view(B, C)    # [B, C]

        # ------------------------------
        # 3) Conjunctive pooling (TP)
        # ------------------------------
        attn_tp = self.attn_head_tp(inst_emb)           # [B, T, 1]
        inst_logits = self.inst_classifier(inst_emb)    # [B, T, C]
        weighted_logits = inst_logits * attn_tp         # [B, T, C]
        bag_logits_tp = weighted_logits.mean(dim=1)     # [B, C]

        if self.is_instance:
            return (
                bag_logits_ts,
                bag_logits_tp,
                weighted_logits,
                inst_logits,
                bag_emb,
                inst_emb,
                attn_ts,
                attn_tp,
            )
        else:
            return bag_logits_ts


class ContextualMIL(nn.Module):
    """
    ContextualMIL (ContextualAmbiguousMILwithCL from expmil.py)

    HybridMIL variant that injects self-attention context into inst_emb
    before contrastive learning and pooling.

    Architecture:
      - InceptionTime backbone
      - Optional Sinusoidal Positional Encoding
      - Self-attention layers for context injection
      - Dual pooling (TS + TP)
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        mDim: int = 128,
        dropout: float = 0.0,
        is_instance: bool = True,
        attn_layers: int = 1,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        use_pos_enc: bool = True,
    ) -> None:
        super().__init__()
        self.is_instance = is_instance
        self.n_classes = n_classes
        self.mDim = mDim
        self.use_pos_enc = use_pos_enc

        out_channels = max(1, mDim // 4)
        self.feature_extractor = InceptionTimeFeatureExtractor(
            n_in_channels=in_features
        )

        self.pos_enc = SinusoidalPositionalEncoding(mDim) if use_pos_enc else None
        self.context_blocks = nn.ModuleList(
            [_SelfAttnBlock(dim=mDim, heads=attn_heads, dropout=attn_dropout) for _ in range(attn_layers)]
        )

        self.attn_head_ts = nn.Sequential(
            nn.Linear(mDim, 8 * n_classes),
            nn.Tanh(),
            nn.Linear(8 * n_classes, n_classes),
            nn.Sigmoid(),
        )
        self.bag_classifier_ts = nn.Linear(mDim, 1)

        self.attn_head_tp = nn.Sequential(
            nn.Linear(mDim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.inst_classifier = nn.Linear(mDim, n_classes)

        self.dropout_p = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, warmup: bool = False):
        """
        Args:
            x: [B, T, D_in]
            warmup: bool (for compatibility)

        Returns:
            Same 8-tuple as HybridMIL
        """
        B, T, D_in = x.shape

        x = x.transpose(1, 2)              # [B, D_in, T]
        feats = self.feature_extractor(x)  # [B, mDim, T]
        if self.dropout_p > 0:
            feats = self.dropout(feats)

        inst_emb = feats.transpose(1, 2)   # [B, T, mDim]
        if self.pos_enc is not None:
            inst_emb = self.pos_enc(inst_emb)

        # Inject global context before heads/CL.
        for block in self.context_blocks:
            inst_emb = block(inst_emb)

        score_ts = self.attn_head_ts(inst_emb)      # [B, T, C]
        score_ts = score_ts.permute(0, 2, 1)        # [B, C, T]
        attn_ts = F.softmax(score_ts, dim=-1)       # [B, C, T]

        bag_emb = torch.bmm(attn_ts, inst_emb)      # [B, C, mDim]
        B, C, D = bag_emb.shape
        bag_emb_flat = bag_emb.reshape(B * C, D)
        bag_logits_flat = self.bag_classifier_ts(bag_emb_flat)
        bag_logits_ts = bag_logits_flat.view(B, C)

        attn_tp = self.attn_head_tp(inst_emb)       # [B, T, 1]
        inst_logits = self.inst_classifier(inst_emb)# [B, T, C]
        weighted_logits = inst_logits * attn_tp
        bag_logits_tp = weighted_logits.mean(dim=1)

        if self.is_instance:
            return (
                bag_logits_ts,
                bag_logits_tp,
                weighted_logits,
                inst_logits,
                bag_emb,
                inst_emb,
                attn_ts,
                attn_tp,
            )
        else:
            return bag_logits_ts
