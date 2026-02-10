from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def build_trace_sequences(meta: pd.DataFrame, cluster_id: np.ndarray) -> dict[tuple[str, str], list[int]]:
    """
    Group by (app_id, trace_id) and produce a cluster sequence per trace.
    Assumes meta has app_id, trace_id, screen_id; and cluster_id aligned with meta rows.

    Ordering:
      - If screen_id is numeric-like, we sort numerically.
      - Else sort lexicographically.
    """
    if len(meta) != len(cluster_id):
        raise ValueError("meta and cluster_id length mismatch")

    df = meta[["app_id", "trace_id", "screen_id"]].copy()
    df["cluster_id"] = cluster_id

    # sort within trace
    def sort_key(x):
        s = str(x)
        return int(s) if s.isdigit() else s

    sequences = {}
    for (app, trace), g in df.groupby(["app_id", "trace_id"]):
        g = g.copy()
        g["__k"] = g["screen_id"].map(sort_key)
        g = g.sort_values("__k")
        sequences[(app, trace)] = g["cluster_id"].tolist()
    return sequences

def bigram_counts(seqs: dict[tuple[str, str], list[int]]) -> Counter[tuple[int, int]]:
    c = Counter()
    for _, s in seqs.items():
        for a, b in zip(s[:-1], s[1:]):
            c[(a, b)] += 1
    return c

def bigram_next_distribution(counts: Counter[tuple[int,int]], num_clusters: int, smoothing: float = 1.0):
    """
    Build P(next | cur) as dense matrix (K x K).
    """
    K = num_clusters
    M = np.full((K, K), smoothing, dtype=np.float32)
    for (a, b), v in counts.items():
        M[a, b] += v
    M /= M.sum(axis=1, keepdims=True)
    return M
