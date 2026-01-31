from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray) -> float:
    """Stable logsumexp for 1D arrays; supports -inf entries."""
    m = float(np.max(x))
    if np.isneginf(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(x - m))))


def softmax_stable(x: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D arrays; supports -inf entries."""
    lse = logsumexp(x)
    if np.isneginf(lse):
        return np.ones_like(x) / float(len(x))
    return np.exp(x - lse)
