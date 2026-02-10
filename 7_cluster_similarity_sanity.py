import numpy as np
import pandas as pd
from tqdm import tqdm
import random

def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

def load_clip(clip_tsv, clip_npy):
    df = pd.read_csv(clip_tsv, sep="\t", low_memory=False)
    X = np.load(clip_npy).astype(np.float32)
    assert len(df) == X.shape[0]
    if "screen_key" not in df.columns:
        if "screen_id" not in df.columns:
            df["screen_id"] = df["image_path"].apply(lambda p: str(p).split("\\")[-1].split("/")[-1].rsplit(".", 1)[0])
        df["screen_key"] = df["app_id"].astype(str) + "::" + df["trace_id"].astype(str) + "::" + df["screen_id"].astype(str)
    return df[["screen_key"]], X

def load_text(text_meta_tsv, text_npy):
    df = pd.read_csv(text_meta_tsv, sep="\t", low_memory=False)
    X = np.load(text_npy).astype(np.float32)
    assert len(df) == X.shape[0]
    if "screen_key" not in df.columns:
        df["screen_key"] = df["app_id"].astype(str) + "::" + df["trace_id"].astype(str) + "::" + df["screen_id"].astype(str)
    return df[["screen_key"]], X

def align_by_key(dfA, XA, dfB, XB):
    keys = list(set(dfA.screen_key) & set(dfB.screen_key))
    A = dfA[dfA.screen_key.isin(keys)].copy()
    B = dfB[dfB.screen_key.isin(keys)].copy()
    XA2 = XA[A.index.values]
    XB2 = XB[B.index.values]
    A = A.reset_index(drop=True)
    B = B.reset_index(drop=True)

    key_to_i = {k:i for i,k in enumerate(A.screen_key.tolist())}
    perm = [key_to_i[k] for k in B.screen_key.tolist()]
    XB2 = XB2[perm]
    B = B.iloc[np.argsort(perm)].reset_index(drop=True)
    assert all(A.screen_key.values == B.screen_key.values)
    return A, XA2, XB2

def sample_pairs_same_cluster(idx_by_cluster, n_pairs, rng):
    pairs = []
    clusters = [c for c, idxs in idx_by_cluster.items() if len(idxs) >= 2]
    if not clusters:
        return pairs
    for _ in range(n_pairs):
        c = rng.choice(clusters)
        a, b = rng.sample(idx_by_cluster[c], 2)
        pairs.append((a, b))
    return pairs

def sample_pairs_diff_cluster(idx_by_cluster, n_pairs, rng):
    pairs = []
    clusters = [c for c, idxs in idx_by_cluster.items() if len(idxs) >= 1]
    if len(clusters) < 2:
        return pairs
    for _ in range(n_pairs):
        c1, c2 = rng.sample(clusters, 2)
        a = rng.choice(idx_by_cluster[c1])
        b = rng.choice(idx_by_cluster[c2])
        pairs.append((a, b))
    return pairs

def compute_cosine_pairs(X, pairs, block=200000):
    # X must be normalized
    sims = []
    for i in range(0, len(pairs), block):
        batch = pairs[i:i+block]
        A = np.array([p[0] for p in batch], dtype=np.int64)
        B = np.array([p[1] for p in batch], dtype=np.int64)
        sims.append(np.sum(X[A] * X[B], axis=1))
    return np.concatenate(sims) if sims else np.array([], dtype=np.float32)

def main(
    #clusters_tsv="processed_data/clusters/screen_clusters.tsv",
    clusters_tsv="processed_data/clusters/screen_clusters_k200.tsv",
    clip_tsv="processed_data/clip/clip_embeddings_with_ids.tsv",
    clip_npy="processed_data/clip/clip_embeddings.npy",
    text_meta_tsv="processed_data/sbert/sbert_text_meta.tsv",
    text_npy="processed_data/sbert/sbert_text_embeddings.npy",
    n_pairs=20000,
    seed=42,
    alpha=1.0,
    beta=1.0
):
    clusters = pd.read_csv(clusters_tsv, sep="\t")
    # clusters must have screen_key and cluster_id
    assert "screen_key" in clusters.columns and "cluster_id" in clusters.columns

    clip_df, clip_X = load_clip(clip_tsv, clip_npy)
    text_df, text_X = load_text(text_meta_tsv, text_npy)

    keys_df, clip_X2, text_X2 = align_by_key(clip_df, clip_X, text_df, text_X)

    # Join cluster_id
    merged = keys_df.merge(clusters[["screen_key", "cluster_id"]], on="screen_key", how="inner")
    keep_idx = merged.index.values
    clip_X2 = clip_X2[keep_idx]
    text_X2 = text_X2[keep_idx]
    cluster_ids = merged["cluster_id"].values

    clip_X2 = l2_normalize(clip_X2)
    text_X2 = l2_normalize(text_X2)

    fused = np.concatenate([alpha * clip_X2, beta * text_X2], axis=1).astype(np.float32)
    fused = l2_normalize(fused)

    # Build indices per cluster
    idx_by_cluster = {}
    for i, c in enumerate(cluster_ids):
        idx_by_cluster.setdefault(int(c), []).append(i)

    rng = random.Random(seed)

    same_pairs = sample_pairs_same_cluster(idx_by_cluster, n_pairs, rng)
    diff_pairs = sample_pairs_diff_cluster(idx_by_cluster, n_pairs, rng)

    same_sims = compute_cosine_pairs(fused, same_pairs)
    diff_sims = compute_cosine_pairs(fused, diff_pairs)

    print(clusters_tsv)
    print("=== Intra vs Inter Cluster Cosine Similarity (FUSED) ===")
    print(f"pairs (same cluster): {len(same_sims)}")
    print(f"pairs (diff cluster): {len(diff_sims)}")
    print(f"intra mean={same_sims.mean():.4f} std={same_sims.std():.4f} median={np.median(same_sims):.4f}")
    print(f"inter mean={diff_sims.mean():.4f} std={diff_sims.std():.4f} median={np.median(diff_sims):.4f}")
    print(f"gap (mean intra - mean inter) = {(same_sims.mean()-diff_sims.mean()):.4f}")

if __name__ == "__main__":
    main()


"""

Quick metric: intra-cluster similarity vs inter-cluster similarity

Sample 1,000 pairs within same cluster → average cosine

Sample 1,000 pairs from different clusters → average cosine
If within >> between, clusters are meaningful.

This is the kind of little plot/table reviewers love because it’s “objective but cheap.”

1) Intra vs inter cluster similarity sanity check
python analysis/07_cluster_similarity_sanity.py
How to interpret:

You want intra mean significantly higher than inter mean.

A healthy gap might be 0.15–0.40 depending on cluster granularity.

If the gap is tiny (<0.05), clustering is not meaningful (not your case, likely).
"""