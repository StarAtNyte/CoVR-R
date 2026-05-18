import numpy as np


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fuse_scores(retrieval: np.ndarray, rerank: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * minmax_norm(retrieval) + (1.0 - alpha) * minmax_norm(rerank)


def combine_dual_sim(primary: np.ndarray, secondary: np.ndarray,
                     weight: float) -> np.ndarray:
    return (1.0 - weight) * primary + weight * secondary
