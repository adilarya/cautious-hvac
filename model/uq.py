"""Layer 3 -- Uncertainty quantification: pocket detection + calibration.

Works on either 2D or 3D arrays. All routines are numpy-shape-agnostic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence
import numpy as np


@dataclass
class PocketMap:
    hot:       np.ndarray   # bool, same shape as mean_field
    cold:      np.ndarray
    uncertain: np.ndarray
    ci_lo:     np.ndarray
    ci_hi:     np.ndarray
    n_hot:        int = 0
    n_cold:       int = 0
    n_uncertain:  int = 0

    def __post_init__(self):
        self.n_hot = int(self.hot.sum())
        self.n_cold = int(self.cold.sum())
        self.n_uncertain = int(self.uncertain.sum())


@dataclass
class CalibrationResult:
    nominal_levels:     np.ndarray
    empirical_coverage: np.ndarray
    ECE:         float
    signed_bias: float
    n_samples:   int


def detect_pockets(
    mean_field: np.ndarray,
    std_field:  np.ndarray,
    setpoint:   float,
    hot_thr:    float = 2.0,
    cold_thr:   float = 2.0,
    unc_thr:    float = 0.8,
) -> PocketMap:
    """Classify each cell. Pockets require both a temperature deviation
    larger than `*_thr` AND posterior std below `unc_thr` (confident)."""
    z = 1.645  # 90% credible-interval z-score
    ci_lo = mean_field - z * std_field
    ci_hi = mean_field + z * std_field

    confident = std_field < unc_thr
    hot  = confident & (mean_field > setpoint + hot_thr)
    cold = confident & (mean_field < setpoint - cold_thr)
    uncertain = ~confident

    return PocketMap(hot=hot, cold=cold, uncertain=uncertain,
                     ci_lo=ci_lo, ci_hi=ci_hi)


def calibration_over_episode(
    mean_history:  Sequence[np.ndarray],
    std_history:   Sequence[np.ndarray],
    truth_history: Sequence[np.ndarray],
    nominal_levels: np.ndarray | None = None,
) -> CalibrationResult:
    """Empirical coverage vs nominal credible-interval level across a
    recorded episode. Pools all (cell, timestep) pairs.

    ECE = mean |empirical - nominal|. Target ECE < 0.05.
    """
    if nominal_levels is None:
        nominal_levels = np.linspace(0.50, 0.95, 10)
    nominal_levels = np.asarray(nominal_levels, dtype=float)

    mean_all  = np.concatenate([np.asarray(m).ravel() for m in mean_history])
    std_all   = np.concatenate([np.asarray(s).ravel() for s in std_history])
    truth_all = np.concatenate([np.asarray(t).ravel() for t in truth_history])
    n = int(len(truth_all))

    from scipy import stats
    coverage = np.zeros_like(nominal_levels)
    for i, alpha in enumerate(nominal_levels):
        z = float(stats.norm.ppf(0.5 + alpha / 2.0))
        lo = mean_all - z * std_all
        hi = mean_all + z * std_all
        coverage[i] = float(np.mean((truth_all >= lo) & (truth_all <= hi)))

    diff = coverage - nominal_levels
    return CalibrationResult(
        nominal_levels=nominal_levels,
        empirical_coverage=coverage,
        ECE=float(np.mean(np.abs(diff))),
        signed_bias=float(np.mean(diff)),
        n_samples=n,
    )
