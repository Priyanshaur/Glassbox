"""Cosine LR schedule with linear warmup. Pure function of step.

No scheduler blob is stored in checkpoints: step + config fully determine
the LR, so resume only needs the integer step. Continuous at the warmup
boundary by construction: warmup ends at peak, cosine starts at peak.
"""
import math


def lr_at(step, peak, warmup_steps, max_steps, min_lr_ratio=0.1):
    """LR for 0-based `step`. step < warmup ramps (step+1)/warmup -> peak."""
    assert 0 <= step < max_steps, "step out of range"
    assert 0 < warmup_steps < max_steps, "bad warmup"
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    p = (step - warmup_steps) / (max_steps - warmup_steps)
    lo = peak * min_lr_ratio
    return lo + 0.5 * (peak - lo) * (1.0 + math.cos(math.pi * p))
