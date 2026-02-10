from __future__ import annotations
import numpy as np

def topk_cosine_neighbors_blockwise(
    X: np.ndarray,   # L2-normalized (N, D)
    q_idx: int,
    k: int = 5,
    block: int = 8192,
) -> list[tuple[int, float]]:
    """
    Memory-safe top-k cosine neighbors without forming NxN.
    """
    q = X[q_idx]
    best_idx = []
    best_sim = []

    N = X.shape[0]
    for start in range(0, N, block):
        end = min(start + block, N)
        sims = X[start:end] @ q
        if start <= q_idx < end:
            sims[q_idx - start] = -np.inf

        if (end - start) > k:
            loc = np.argpartition(-sims, k)[:k]
        else:
            loc = np.arange(end - start)

        for j in loc:
            best_idx.append(start + int(j))
            best_sim.append(float(sims[int(j)]))

    best_idx = np.array(best_idx, dtype=np.int64)
    best_sim = np.array(best_sim, dtype=np.float32)
    top = np.argsort(-best_sim)[:k]
    return [(int(best_idx[i]), float(best_sim[i])) for i in top]
