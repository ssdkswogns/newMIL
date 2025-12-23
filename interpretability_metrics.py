"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Multi-label compatible adaptation of
https://github.com/JAEarly/MILTimeSeriesClassification/blob/master/millet/interpretability_metrics.py
"""
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from millet.model.millet_model import MILLETModel
from millet.util import custom_tqdm


def _resolve_target_classes(
    bag_logits: torch.Tensor,
    bag_labels: Optional[torch.Tensor],
    target_classes: Optional[List[List[int]]],
    threshold: float,
    fallback_to_predicted: bool,
) -> List[Tuple[int, int]]:
    """
    Build a flat list of (bag_idx, class_idx) pairs to evaluate.
    Priority:
      1) target_classes if given
      2) positive classes from bag_labels (multi-hot) using threshold
      3) predicted argmax class (for backward compatibility)
    """
    n_bags, n_classes = bag_logits.shape
    targets: List[Tuple[int, int]] = []

    if target_classes is not None:
        for b, clzs in enumerate(target_classes):
            for c in clzs:
                targets.append((b, int(c)))
        if targets:
            return targets

    if bag_labels is not None:
        if bag_labels.shape[0] != n_bags:
            raise ValueError("bag_labels batch size does not match bag_logits")
        pos = (bag_labels > threshold).nonzero(as_tuple=False)
        for b, c in pos:
            targets.append((int(b.item()), int(c.item())))
        if targets:
            return targets

    if not fallback_to_predicted:
        return targets

    pred_clzs = torch.argmax(bag_logits, dim=1)
    for b in range(n_bags):
        targets.append((b, int(pred_clzs[b].item())))
    return targets


def calculate_aopcr(
    model: "MILLETModel",
    bags: List[torch.Tensor],
    verbose: bool = True,
    stop: float = 0.5,
    step: float = 0.05,
    n_random: int = 3,
    seed: int = 72,
    batch_interpretations: Optional[torch.Tensor] = None,
    bag_labels: Optional[torch.Tensor] = None,
    target_classes: Optional[List[List[int]]] = None,
    label_threshold: float = 0.5,
    fallback_to_predicted: bool = True,
    return_targets: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the Area Over the Perturbation Curve to Random (AOPCR).

    Multi-label support:
      - Provide `bag_labels` (multi-hot) to evaluate every positive class.
      - Or pass `target_classes` as a list of class indices per bag.
      - If neither is given, falls back to the original single-class behaviour
        by using the predicted argmax class per bag.

    Returns AOPCR/curves stacked over target pairs in the order returned in
    `target_pairs` (if `return_targets=True`).
    """
    # Get model output for bags
    batch_model_out = model(bags)
    batch_bag_logits = batch_model_out["bag_logits"]
    if batch_bag_logits.dim() != 2:
        raise ValueError("bag_logits must be 2D [batch, classes]")
    batch_bag_logits_cpu = batch_bag_logits.detach().cpu()

    target_pairs = _resolve_target_classes(
        batch_bag_logits_cpu,
        bag_labels.detach().cpu() if bag_labels is not None else None,
        target_classes,
        label_threshold,
        fallback_to_predicted,
    )
    if not target_pairs:
        raise ValueError("No target classes found for AOPCR computation")

    # Gather original logits for each (bag, class) pair
    batch_orig_logits = torch.tensor(
        [batch_bag_logits_cpu[b, c].item() for b, c in target_pairs]
    )

    # Interpretations for each target pair
    if batch_interpretations is None:
        all_batch_interpretations = model.interpret(batch_model_out)
    else:
        all_batch_interpretations = batch_interpretations

    if all_batch_interpretations.dim() != 3:
        raise ValueError("Interpretations must be 3D [batch, classes, instances]")

    interp_per_target = torch.stack(
        [all_batch_interpretations[b, c, :] for b, c in target_pairs]
    )

    # Build per-target bags and class list
    bags_per_target = [bags[b] for b, _ in target_pairs]
    clzs_per_target = [c for _, c in target_pairs]

    # Calculate AOPC for given interpretations
    aopc, pc = _calculate_aopc(
        model,
        bags_per_target,
        batch_orig_logits,
        clzs_per_target,
        interp_per_target,
        stop,
        step,
        verbose,
    )
    # Calculate AOPC for random orderings
    r_aopc, r_pc = _calculate_random_aopc(
        model,
        bags_per_target,
        batch_orig_logits,
        clzs_per_target,
        stop,
        step,
        verbose,
        n_random,
        seed,
    )
    aopcr = aopc - r_aopc

    if return_targets:
        return aopcr, pc, r_pc, target_pairs
    return aopcr, pc, r_pc


def _calculate_aopc(
    model: "MILLETModel",
    bags: List[torch.Tensor],
    batch_orig_logits: torch.Tensor,
    clzs: List[int],
    batch_interpretation_scores: torch.Tensor,
    stop: float,
    step: float,
    verbose: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Area Over the Perturbation Curve
    """
    # Compute perturbation steps
    n_targets = len(bags)
    n_instances = len(bags[0])
    steps = np.linspace(1, stop, num=math.ceil((1 - stop) / step + 1))
    n_steps = len(steps)
    # Set up perturbation curve and fill first row with original logits
    batch_pc = torch.zeros((n_targets, n_steps))
    batch_pc[:, 0] = batch_orig_logits
    # Get the instance orderings by most relevant (greatest score) first.
    batch_morf = torch.argsort(batch_interpretation_scores.cpu(), descending=True, stable=True)
    # Actually compute the perturbation curve
    for step_idx, step in custom_tqdm(
        enumerate(steps[1:]),
        total=n_steps - 1,
        desc="Computing perturbation curve",
        leave=False,
        disable=not verbose,
    ):
        # Work out how many instances to remove
        n_to_remove = int((1 - step) * n_instances)
        # Create perturbed bags and their respective positions
        perturbed_bags = []
        perturbed_bags_pos = []
        for i in range(n_targets):
            b, p = _create_perturbed_bag(bags[i], batch_morf[i], n_to_remove)
            perturbed_bags.append(b)
            perturbed_bags_pos.append(p)
        # Pass perturbed bags through the model to get the new logits
        with torch.no_grad():
            new_logits = model(perturbed_bags, torch.stack(perturbed_bags_pos))["bag_logits"]
        # Update output
        for target_idx, clz in enumerate(clzs):
            batch_pc[target_idx, step_idx + 1] = new_logits[target_idx, clz].item()
    # Compute the AOPC for each target in the batch
    batch_aopc = torch.zeros(n_targets)
    for k in range(1, n_steps):
        batch_aopc += batch_pc[:, 0] - batch_pc[:, k]
    batch_aopc /= n_steps
    # Adjust the perturbation curves to start at 0
    batch_pc -= batch_orig_logits.unsqueeze(1)
    return batch_aopc, batch_pc


def _calculate_random_aopc(
    model: "MILLETModel",
    bags: List[torch.Tensor],
    batch_orig_logits: torch.Tensor,
    clzs: List[int],
    stop: float,
    step: float,
    verbose: bool,
    n_repeats: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Area of the Perturbation Curve for random orderings.
    """
    n_targets = len(bags)
    n_instances = len(bags[0])
    torch.random.manual_seed(seed)
    random_aopcs = []
    random_pcs = []
    for _ in range(n_repeats):
        # Create random interpretation scores (random ordering)
        random_interpretation_scores = torch.rand((n_targets, n_instances))
        # Compute AOPC and PC for this random ordering
        r_aopc, r_pc = _calculate_aopc(
            model,
            bags,
            batch_orig_logits,
            clzs,
            random_interpretation_scores,
            stop,
            step,
            verbose,
        )
        random_aopcs.append(r_aopc)
        random_pcs.append(r_pc)
    # Compute average over repeats
    aopc = torch.stack(random_aopcs).mean(dim=0)
    pc = torch.stack(random_pcs).mean(dim=0)
    return aopc, pc


def _create_perturbed_bag(
    bag: torch.Tensor, bag_morf: torch.Tensor, n_to_remove: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perturb a bag by removing the n most important instances.
    """
    # Create mask of indices to remove
    mask = torch.ones(len(bag), dtype=torch.int)
    idxs_to_remove = bag_morf[:n_to_remove]
    mask[idxs_to_remove] = 0
    # Remove based on mask and create positions
    perturbed_bag = bag[mask == 1]
    perturbed_bag_pos = torch.arange(len(bag))[mask == 1]
    return perturbed_bag, perturbed_bag_pos


def calculate_ndcg_at_n(instance_importance_scores: torch.Tensor, instance_labels: torch.Tensor) -> float:
    """
    Calculate Normalised Discounted Cumulative Gain @ n (NDCG@n).
    Evaluation of MIL interpretability that requires instance labels.
    """
    # Identify number of discriminatory instances
    n = int((instance_labels == 1).sum().item())
    # No targets so return nan
    if n == 0:
        raise ValueError("Trying to assess interpretability with no discriminatory instances")
    # Find idxs of the n largest interpretation scores
    top_n = torch.topk(instance_importance_scores.to("cpu"), n)[1]
    # Compute normalised discounted cumulative gain
    dcg = 0.0
    norm = 0.0
    for i, order_idx in enumerate(top_n):
        rel = instance_labels[order_idx].item()
        dcg += rel / math.log2(i + 2)
        norm += 1.0 / math.log2(i + 2)
    ndcg = dcg / norm
    return ndcg
