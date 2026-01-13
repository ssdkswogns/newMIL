# Extended AOPCR Evaluation Guide

## Overview

The AOPCR (Area Over the Perturbation Curve to Random) metric has been extended to support:

1. **Full range perturbation** (0% to 100%, not just 0-50%)
2. **Coverage metrics** at specific performance thresholds
3. **Flexible configuration** via command-line arguments

## Key Improvements

### 1. Full Range Support

**Before:** Only measured up to 50% perturbation
```python
# Old: stop=0.5 (hardcoded)
aopcr = compute_classwise_aopcr(model, testloader, args, stop=0.5)
```

**After:** Configurable from 0% to 100%
```bash
# Evaluate with full range (0-100%)
python eval.py --model AmbiguousMIL --aopcr_stop 1.0 --aopcr_step 0.05

# Traditional evaluation (0-50%)
python eval.py --model AmbiguousMIL --aopcr_stop 0.5 --aopcr_step 0.05

# More granular steps (1% increments)
python eval.py --model AmbiguousMIL --aopcr_stop 1.0 --aopcr_step 0.01
```

### 2. Coverage Metrics

**New metric:** Coverage@X = % of instances needed to maintain X% of original performance

Example output:
```
===== Coverage Metrics =====
(Coverage@X = % of instances needed to maintain X% of original performance)

Coverage@90%:
  Explanation: 0.1234 (weighted: 0.1189)
  Random:      0.2456 (weighted: 0.2401)
  Gain:        -0.1222 (explanation better than random)

Coverage@80%:
  Explanation: 0.2345 (weighted: 0.2301)
  Random:      0.3567 (weighted: 0.3512)
  Gain:        -0.1222 (explanation better than random)
```

**Interpretation:**
- **Lower is better**: Need fewer instances to maintain performance
- **Negative gain**: Explanation-based selection is more efficient than random
- Shows how well the model identifies truly important instances

### 3. Parameter Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--aopcr_stop` | 0.5 | Perturbation limit (0.0-1.0). Use 1.0 for full range. |
| `--aopcr_step` | 0.05 | Perturbation step size (e.g., 0.05 = 5% increments) |
| `--num_random` | 3 | Number of random baseline repetitions |

## Usage Examples

### Example 1: Compare Traditional vs Full Range

```bash
# Traditional MIL evaluation (0-50%)
python eval.py \
    --model TimeMIL \
    --model_path savemodel/InceptBackbone/BasicMotions/exp_1/weights/best_TimeMIL.pth \
    --dataset BasicMotions \
    --aopcr_stop 0.5 \
    --aopcr_step 0.05

# AmbiguousMIL with CL evaluation (0-100%)
python eval.py \
    --model AmbiguousMIL \
    --model_path savemodel/InceptBackbone/BasicMotions/exp_2/weights/best_AmbiguousMIL.pth \
    --dataset BasicMotions \
    --aopcr_stop 1.0 \
    --aopcr_step 0.05
```

### Example 2: High-Resolution Analysis

```bash
# Fine-grained analysis with 1% steps
python eval.py \
    --model AmbiguousMIL \
    --model_path savemodel/InceptBackbone/BasicMotions/exp_2/weights/best_AmbiguousMIL.pth \
    --dataset BasicMotions \
    --aopcr_stop 1.0 \
    --aopcr_step 0.01 \
    --num_random 10
```

### Example 3: Collect Results Across Datasets

```bash
# Run evaluations
for dataset in BasicMotions Heartbeat NATOPS; do
    python eval.py \
        --model AmbiguousMIL \
        --model_path savemodel/InceptBackbone/${dataset}/exp_latest/weights/best_AmbiguousMIL.pth \
        --dataset $dataset \
        --aopcr_stop 1.0 \
        --aopcr_step 0.05
done

# Collect and summarize
python collect_aopcr.py --root savemodel/InceptBackbone
```

## Expected Results

### Traditional MIL (TimeMIL, BasicMIL)
- **AOPCR@0.5**: High (good at finding key instances)
- **AOPCR@1.0**: Moderate (remaining 50% less useful)
- **Coverage@90%**: ~0.15-0.20 (need 15-20% of instances)

### AmbiguousMIL with CL
- **AOPCR@0.5**: High (also good at finding key instances)
- **AOPCR@1.0**: **Higher** (better utilization across entire sequence)
- **Coverage@90%**: **~0.10-0.15** (more efficient instance selection)

## Research Story

> "Our model with Contrastive Learning not only identifies key instances effectively (similar AOPCR@0.5), but also leverages ambiguous instances across the entire temporal sequence more efficiently (higher AOPCR@1.0 and better coverage metrics), demonstrating superior utilization of temporal information."

## Implementation Details

### Return Values

```python
result = compute_classwise_aopcr(
    model, testloader, args,
    stop=1.0,
    step=0.05,
    n_random=3,
    coverage_thresholds=[0.9, 0.8, 0.7, 0.5]
)

aopcr_per_class,    # (C,) array: AOPCR for each class
aopcr_weighted,     # scalar: weighted average AOPCR
aopcr_mean,         # scalar: simple mean AOPCR
aopcr_overall,      # scalar: overall AOPCR (original dataset only)
M_expl,             # (C, n_steps) array: explanation curves
M_rand,             # (C, n_steps) array: random curves
alphas,             # (n_steps,) array: perturbation ratios
counts,             # (C,) array: bag counts per class
coverage_summary    # dict: coverage metrics
    = result
```

### Coverage Summary Structure

```python
coverage_summary = {
    0.9: {  # Maintain 90% performance
        'expl_per_class': np.array([...]),    # Per-class coverage
        'rand_per_class': np.array([...]),
        'expl_weighted': 0.1234,              # Weighted average
        'rand_weighted': 0.2456,
        'expl_mean': 0.1189,                  # Simple mean
        'rand_mean': 0.2401,
        'coverage_gain': -0.1222,             # Expl - Rand (negative = better)
    },
    # ... other thresholds
}
```

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce step size or evaluate in smaller batches
```bash
python eval.py --aopcr_stop 1.0 --aopcr_step 0.1 --batchsize 32
```

### Issue: Coverage metrics show NaN
**Cause:** Model performance drops too quickly
**Solution:** Check if model is properly trained or use lower thresholds
```python
coverage_thresholds=[0.7, 0.5, 0.3]  # Lower thresholds
```

### Issue: Very long computation time
**Solution:** Use fewer random repetitions or larger steps
```bash
python eval.py --aopcr_stop 1.0 --aopcr_step 0.1 --num_random 1
```

## References

- Original AOPCR: [MILLET Paper](https://github.com/JAEarly/MILTimeSeriesClassification)
- Extended implementation: `compute_aopcr.py`
- Evaluation script: `eval.py`
- Collection script: `collect_aopcr.py`
