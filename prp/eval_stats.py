"""
Statistical validation for evaluation: bootstrap 95% CI, paired comparison (bootstrap difference or Wilcoxon).
"""
import random
from typing import List, Tuple

import numpy as np


def bootstrap_ci(
    values: List[float],
    n: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.
    Returns (mean, lo, hi).
    """
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    m = arr.mean()
    size = len(arr)
    rng = random.Random(42)
    boot_means = []
    for _ in range(n):
        idx = rng.choices(range(size), k=size)
        boot_means.append(arr[idx].mean())
    boot_means.sort()
    lo_idx = int(alpha / 2 * n)
    hi_idx = int((1 - alpha / 2) * n)
    lo = boot_means[lo_idx] if lo_idx < n else boot_means[0]
    hi = boot_means[hi_idx] if hi_idx < n else boot_means[-1]
    return (float(m), float(lo), float(hi))


def paired_bootstrap_diff(
    values_a: List[float],
    values_b: List[float],
    n: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap: difference = B - A. Returns (diff_mean, ci_lo, ci_hi).
    Pairs by index; shorter list is padded with NaN and those pairs excluded, or truncate to min len.
    """
    na, nb = len(values_a), len(values_b)
    if na == 0 or nb == 0:
        return (0.0, 0.0, 0.0)
    size = min(na, nb)
    a = np.asarray(values_a[:size], dtype=float)
    b = np.asarray(values_b[:size], dtype=float)
    diff = b - a
    diff_mean = float(diff.mean())
    rng = random.Random(42)
    boot_diffs = []
    for _ in range(n):
        idx = rng.choices(range(size), k=size)
        boot_diffs.append(diff[idx].mean())
    boot_diffs.sort()
    lo_idx = int(alpha / 2 * n)
    hi_idx = int((1 - alpha / 2) * n)
    ci_lo = boot_diffs[lo_idx] if lo_idx < n else boot_diffs[0]
    ci_hi = boot_diffs[hi_idx] if hi_idx < n else boot_diffs[-1]
    return (diff_mean, float(ci_lo), float(ci_hi))


def wilcoxon_signed_rank_if_available(values_a: List[float], values_b: List[float]) -> Tuple[bool, float]:
    """
    If scipy is available, run Wilcoxon signed-rank test. Returns (available, p_value).
    """
    try:
        from scipy.stats import wilcoxon
        size = min(len(values_a), len(values_b))
        if size < 3:
            return (True, 1.0)
        stat, p = wilcoxon(values_a[:size], values_b[:size], alternative="two-sided")
        return (True, float(p))
    except ImportError:
        return (False, float("nan"))
