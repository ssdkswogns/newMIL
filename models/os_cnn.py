"""
OS-CNN (Omni-Scale CNN) implementation adapted for this codebase.
Accepts input shaped [B, T, C] and returns logits [B, n_class] plus
an interpretation map (CAM) for timestep importance.
"""
import math
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_mask_index(kernel_length_now: int, largest_kernel_length: int) -> Tuple[int, int]:
    right_zero = math.ceil((largest_kernel_length - 1) / 2) - math.ceil((kernel_length_now - 1) / 2)
    left_zero = largest_kernel_length - kernel_length_now - right_zero
    return left_zero, left_zero + kernel_length_now


def creat_mask(in_channels: int, out_channels: int, kernel_length_now: int, largest_kernel_length: int) -> np.ndarray:
    ind_left, ind_right = calculate_mask_index(kernel_length_now, largest_kernel_length)
    mask = np.ones((out_channels, in_channels, largest_kernel_length))
    mask[:, :, 0:ind_left] = 0
    mask[:, :, ind_right:] = 0
    return mask


def creak_layer_mask(layer_parameter_list: List[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    largest_kernel_length = max(p[2] for p in layer_parameter_list)
    mask_list = []
    init_weight_list = []
    bias_list = []
    for in_c, out_c, k in layer_parameter_list:
        conv = nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=k)
        ind_l, ind_r = calculate_mask_index(k, largest_kernel_length)
        big_weight = np.zeros((out_c, in_c, largest_kernel_length))
        big_weight[:, :, ind_l:ind_r] = conv.weight.detach().numpy()

        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)

        mask = creat_mask(in_c, out_c, k, largest_kernel_length)
        mask_list.append(mask)

    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class OSBlock(nn.Module):
    """One OS-CNN block with multiple kernel sizes merged."""
    def __init__(self, layer_parameters: List[Tuple[int, int, int]]):
        super().__init__()
        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)

        in_channels = os_mask.shape[1]
        out_channels = os_mask.shape[0]
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask), requires_grad=False)
        self.padding = nn.ConstantPad1d((int((max_kernel_size - 1) / 2), int(max_kernel_size / 2)), 0)

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight), requires_grad=True)
        self.conv1d.bias = nn.Parameter(torch.from_numpy(init_bias), requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conv1d.weight.data = self.conv1d.weight * self.weight_mask  # mask kernels
        x = self.padding(x)
        x = self.conv1d(x)
        x = self.bn(x)
        return F.relu(x)


class OS_CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_class: int,
        layer_parameter_list: Optional[List[List[Tuple[int, int, int]]]] = None,
        few_shot: bool = False,
    ):
        """
        in_channels: feature dimension (C)
        n_class: number of classes
        layer_parameter_list: list of blocks; each block is list of (in_c, out_c, kernel)
        few_shot: if True, returns embedding only; else returns logits
        """
        super().__init__()
        self.few_shot = few_shot

        if layer_parameter_list is None:
            # Default two blocks, multi-kernel sizes; ensure channel alignment
            block1 = [
                (in_channels, 32, 7),
                (in_channels, 64, 5),
                (in_channels, 32, 3),
            ]  # sum out = 128
            block2_in = sum(p[1] for p in block1)
            block2 = [
                (block2_in, 64, 5),
                (block2_in, 64, 3),
            ]  # sum out = 128
            layer_parameter_list = [block1, block2]

        self.blocks = nn.ModuleList([OSBlock(params) for params in layer_parameter_list])
        self.averagepool = nn.AdaptiveAvgPool1d(1)

        last_out_channels = sum(p[1] for p in layer_parameter_list[-1])
        self.hidden = nn.Linear(last_out_channels, n_class)

    def forward(self, x: torch.Tensor, return_cam: bool = False):
        """
        x: [B, T, C] -> permuted to [B, C, T]
        returns logits (and optional CAM [B, C, T])
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        for block in self.blocks:
            x = block(x)  # keep shape [B, C_out, T]

        feat_map = x  # [B, C_out, T]
        pooled = self.averagepool(feat_map).squeeze(-1)  # [B, C_out]

        if self.few_shot:
            return pooled

        logits = self.hidden(pooled)

        if not return_cam:
            return logits

        # Class activation map using linear weights
        w = self.hidden.weight  # [n_class, C_out]
        cam = torch.einsum("cf,bft->bct", w, feat_map)
        return logits, cam
