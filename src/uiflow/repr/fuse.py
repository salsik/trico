from __future__ import annotations
import numpy as np
from ..io.load import l2_normalize

def concat_fuse(
    v: np.ndarray,
    t: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Late fusion by concatenation. Usually the best default.
    v: (N, Dv), t: (N, Dt)
    """
    if v.shape[0] != t.shape[0]:
        raise ValueError("v and t must have same N")
    z = np.concatenate([alpha * v, beta * t], axis=1)
    return l2_normalize(z) if normalize else z

def weighted_sum_fuse(
    v: np.ndarray,
    t: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """
    Only valid if v and t have same dimensionality (after projection).
    """
    if v.shape != t.shape:
        raise ValueError("weighted_sum_fuse requires v and t to have the same shape")
    z = alpha * v + beta * t
    return l2_normalize(z) if normalize else z

def similarity_fuse(
    sim_v: np.ndarray,
    sim_t: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Fuse similarities, not vectors:
      sim = alpha*sim_v + (1-alpha)*sim_t
    """
    if sim_v.shape != sim_t.shape:
        raise ValueError("sim matrices must match")
    return alpha * sim_v + (1 - alpha) * sim_t
