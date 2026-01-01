"""
MILLET model (standalone) ported from MILTimeSeriesClassification.
Keeps the original pooling variants and exposes outputs compatible with main_cl_fix.py.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inceptiontime import InceptionTimeFeatureExtractor


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Adapted from (under BSD 3-Clause License):
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor, x_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_pos is None:
            x_pe = self.pe[:, : x.size(1)]
        else:
            x_pe = self.pe[0, x_pos]
        x = x + x_pe
        return x


# ============================================================================
# MIL Pooling Methods
# ============================================================================

class MILPooling(nn.Module):
    """Base class for MIL pooling methods."""

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_classes = n_classes
        self.dropout_p = dropout
        self.apply_positional_encoding = apply_positional_encoding
        if apply_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_in)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)


class GlobalAveragePooling(MILPooling):
    """GAP (EmbeddingSpace MIL) pooling."""

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        dropout: float = 0.0,
        apply_positional_encoding: bool = False,
    ) -> None:
        super().__init__(
            d_in,
            n_classes,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.bag_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: [B, D, T]
        :param pos: Optional positions per timestep
        """
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        instance_embeddings = instance_embeddings.transpose(2, 1)

        cam = self.bag_classifier.weight @ instance_embeddings
        bag_embeddings = instance_embeddings.mean(dim=-1)
        bag_logits = self.bag_classifier(bag_embeddings)
        return {
            "bag_logits": bag_logits,
            "interpretation": cam,
        }


class MILInstancePooling(MILPooling):
    """Instance MIL pooling. Instance prediction then averaging."""

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ) -> None:
        super().__init__(
            d_in,
            n_classes,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.instance_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        instance_logits = self.instance_classifier(instance_embeddings)
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": instance_logits.transpose(1, 2),
            "instance_logits": instance_logits.transpose(1, 2),
        }


class MILAttentionPooling(MILPooling):
    """Attention MIL pooling. Instance attention then weighted averaging of embeddings."""

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ) -> None:
        super().__init__(
            d_in,
            n_classes,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.bag_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        attn = self.attention_head(instance_embeddings)
        instance_embeddings = instance_embeddings * attn
        bag_embedding = torch.mean(instance_embeddings, dim=1)
        bag_logits = self.bag_classifier(bag_embedding)
        return {
            "bag_logits": bag_logits,
            "interpretation": attn.repeat(1, 1, self.n_classes).transpose(1, 2),
            "attn": attn,
        }


class MILAdditivePooling(MILPooling):
    """Additive MIL pooling. Instance attention then weighting of embeddings before instance prediction."""

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ) -> None:
        super().__init__(
            d_in,
            n_classes,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        attn = self.attention_head(instance_embeddings)
        instance_embeddings = instance_embeddings * attn
        instance_logits = self.instance_classifier(instance_embeddings)
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": (instance_logits * attn).transpose(1, 2),
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": attn,
        }


class MILConjunctivePooling(MILPooling):
    """Conjunctive MIL pooling. Instance attention then weighting of instance predictions."""

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ) -> None:
        super().__init__(
            d_in,
            n_classes,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_classes)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        attn = self.attention_head(instance_embeddings)
        instance_logits = self.instance_classifier(instance_embeddings)
        weighted_instance_logits = instance_logits * attn
        bag_logits = torch.mean(weighted_instance_logits, dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": weighted_instance_logits.transpose(1, 2),
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": attn,
        }


# ============================================================================
# MILLET Main Model
# ============================================================================

class MILLET(nn.Module):
    """
    MILLET: Multiple Instance Learning for long-horizon time series classification.
    This implementation is independent from TimeMIL and aligned with MILTimeSeriesClassification.
    """

    def __init__(
        self,
        feats_size: int,
        mDim: int = 128,
        n_classes: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 256,  # unused, kept for interface parity
        pooling: str = "conjunctive",
        is_instance: bool = False,
        apply_positional_encoding: bool = True,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.is_instance = is_instance
        self.n_classes = n_classes
        self.mDim = mDim

        # Feature extractor (InceptionTime backbone)
        out_channels = max(1, mDim // 4) # 그냥 출력 맞춰주기 위해
        self.feature_extractor = InceptionTimeFeatureExtractor(
            n_in_channels=feats_size,
            out_channels=out_channels,
            padding_mode=padding_mode,
        )
        backbone_out_dim = out_channels * 4

        # Optional projection so that backbone output matches mDim
        if backbone_out_dim != mDim:
            self.feature_proj = nn.Conv1d(backbone_out_dim, mDim, kernel_size=1)
        else:
            self.feature_proj = nn.Identity()
        d_in = mDim

        pooling = pooling.lower()
        if pooling == "conjunctive":
            self.pooling = MILConjunctivePooling(
                d_in, n_classes, dropout=dropout,
                apply_positional_encoding=apply_positional_encoding,
            )
        elif pooling == "attention":
            self.pooling = MILAttentionPooling(
                d_in, n_classes, dropout=dropout,
                apply_positional_encoding=apply_positional_encoding,
            )
        elif pooling == "instance":
            self.pooling = MILInstancePooling(
                d_in, n_classes, dropout=dropout,
                apply_positional_encoding=apply_positional_encoding,
            )
        elif pooling == "additive":
            self.pooling = MILAdditivePooling(
                d_in, n_classes, dropout=dropout,
                apply_positional_encoding=apply_positional_encoding,
            )
        elif pooling in ("gap", "embedding", "embeddingspace"):
            self.pooling = GlobalAveragePooling(
                d_in, n_classes, dropout=0.0,
                apply_positional_encoding=False,
            )
        else:
            raise ValueError(f"Pooling {pooling} not supported. Choose from "
                             f"'conjunctive', 'attention', 'instance', 'additive', 'gap'.")

    def forward(self, x: torch.Tensor, warmup: bool = False):
        """
        Args:
            x: [B, T, D] input sequences
            warmup: unused, kept for API compatibility

        Returns:
            - if is_instance=True:
                (bag_logits, instance_pred, x_cls, x_seq, interpretation, attn1, attn2)
            - else if pooling provides instance/interpretation:
                (bag_logits, instance_pred, interpretation)
            - else:
                bag_logits
        """
        # Feature extraction
        x = x.transpose(1, 2)  # [B, D, T]
        features = self.feature_extractor(x)  # [B, ?, T]
        features = self.feature_proj(features)  # [B, mDim, T]

        # Pooling
        pool_out = self.pooling(features)
        bag_logits = pool_out["bag_logits"]  # [B, C]
        instance_logits = pool_out.get("instance_logits", None)  # [B, C, T]
        interpretation = pool_out.get("interpretation", None)  # [B, C, T] or [B, C, ?]

        if self.is_instance:
            instance_pred = instance_logits.transpose(1, 2) if instance_logits is not None else None
            return bag_logits, instance_pred, interpretation

        return bag_logits