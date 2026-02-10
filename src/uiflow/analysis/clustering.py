from __future__ import annotations
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def kmeans_cluster(
    X: np.ndarray,
    k: int,
    seed: int = 42,
    batch_size: int = 4096,
    n_init: int = 10,
) -> np.ndarray:
    """
    Fast scalable baseline clustering for large N.
    Returns cluster_id (N,)
    """
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=batch_size,
        n_init=n_init,
    )
    return km.fit_predict(X)
