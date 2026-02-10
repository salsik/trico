import os
import numpy as np
import pandas as pd
import random
from sklearn.cluster import MiniBatchKMeans


def ensure_screen_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has screen_id. If missing, derive from image_path or json_path filename stem.
    """
    if "screen_id" in df.columns:
        return df

    if "image_path" in df.columns:
        df["screen_id"] = df["image_path"].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
        return df

    if "json_path" in df.columns:
        df["screen_id"] = df["json_path"].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
        return df

    raise KeyError("No screen_id column and neither image_path nor json_path exists to derive it.")


def ensure_required_cols(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"app_id", "trace_id"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} TSV missing required columns: {missing}. Columns are: {list(df.columns)}")
    df = ensure_screen_id(df)
    return df


def build_screen_key(df: pd.DataFrame) -> pd.Series:
    return df["app_id"].astype(str) + "::" + df["trace_id"].astype(str) + "::" + df["screen_id"].astype(str)


def load_embeddings(tsv_path: str, npy_path: str, dim: int, name: str):
    df = pd.read_csv(tsv_path, sep="\t")
    df = ensure_required_cols(df, name=name)

    embs = np.load(npy_path)
    if embs.shape[0] != len(df):
        raise ValueError(f"{name}: npy rows {embs.shape[0]} != TSV rows {len(df)}")
    if embs.shape[1] != dim:
        raise ValueError(f"{name}: expected dim {dim} but got {embs.shape[1]}")

    df["screen_key"] = build_screen_key(df)
    return df, embs.astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def topk_cosine_neighbors(
    X: np.ndarray,            # normalized (N, D)
    q_idx: int,
    k: int = 5,
    block: int = 8192
):
    """
    Compute top-k cosine neighbors for query index q_idx without NxN matrix.
    Returns list of (idx, sim) sorted by sim desc, excluding itself.
    """
    q = X[q_idx]  # (D,)
    best_idx = []
    best_sim = []

    N = X.shape[0]
    for start in range(0, N, block):
        end = min(start + block, N)
        sims = X[start:end] @ q  # (block,)
        # exclude self if within this block
        if start <= q_idx < end:
            sims[q_idx - start] = -np.inf

        # take local top candidates
        if (end - start) > k:
            loc = np.argpartition(-sims, k)[:k]
        else:
            loc = np.arange(end - start)

        for j in loc:
            best_idx.append(start + int(j))
            best_sim.append(float(sims[int(j)]))

    # global top-k
    best_idx = np.array(best_idx, dtype=np.int64)
    best_sim = np.array(best_sim, dtype=np.float32)
    top = np.argsort(-best_sim)[:k]
    return [(int(best_idx[i]), float(best_sim[i])) for i in top]


def main():
    clip_tsv = "clip_embeddings_with_ids.tsv"
    clip_npy = "clip_embeddings.npy"
    text_tsv = "sbert_text_meta.tsv"
    text_npy = "sbert_text_embeddings.npy"

    clip_df, clip_emb = load_embeddings(clip_tsv, clip_npy, dim=768, name="CLIP")
    text_df, text_emb = load_embeddings(text_tsv, text_npy, dim=384, name="TEXT")

    print(f"CLIP rows: {len(clip_df)} | cols: {list(clip_df.columns)}")
    print(f"TEXT rows: {len(text_df)} | cols: {list(text_df.columns)}")

    # Intersection on screen_key
    common = set(clip_df["screen_key"]) & set(text_df["screen_key"])
    print(f"Intersection (screens with both modalities): {len(common)}")

    clip_mask = clip_df["screen_key"].isin(common).values
    text_mask = text_df["screen_key"].isin(common).values

    clip_df2 = clip_df.loc[clip_mask].copy()
    text_df2 = text_df.loc[text_mask].copy()
    clip_emb2 = clip_emb[clip_mask]
    text_emb2 = text_emb[text_mask]

    # Align TEXT ordering to CLIP by screen_key
    clip_df2 = clip_df2.reset_index(drop=True)
    text_df2 = text_df2.reset_index(drop=True)

    key_to_clip_idx = {k: i for i, k in enumerate(clip_df2["screen_key"].tolist())}
    text_perm = [key_to_clip_idx[k] for k in text_df2["screen_key"].tolist()]
    text_emb2 = text_emb2[text_perm]
    text_df2 = text_df2.iloc[np.argsort(text_perm)].reset_index(drop=True)

    assert all(clip_df2["screen_key"].values == text_df2["screen_key"].values)

    # Normalize
    clip_emb2 = l2_normalize(clip_emb2)
    text_emb2 = l2_normalize(text_emb2)

    # Late fusion (vector concat) for clustering
    fused_vec = np.concatenate([clip_emb2, text_emb2], axis=1)  # (N, 1152)
    fused_vec = l2_normalize(fused_vec)

    # ---- neighbor sanity check (no NxN) ----
    print("\n=== Neighbor sanity check (top-5) ===")
    alpha = 0.5  # similarity fusion weight
    picks = random.sample(range(len(clip_df2)), k=min(5, len(clip_df2)))

    for qi in picks:
        qkey = clip_df2.iloc[qi]["screen_key"]
        print("\n---")
        print(f"QUERY: {qkey}")

        # neighbors using CLIP only
        clip_nbrs = topk_cosine_neighbors(clip_emb2, qi, k=5)
        print("CLIP neighbors:")
        for j, s in clip_nbrs:
            print(f"  {clip_df2.iloc[j]['screen_key']}  sim={s:.4f}")

        # neighbors using TEXT only
        text_nbrs = topk_cosine_neighbors(text_emb2, qi, k=5)
        print("TEXT neighbors:")
        for j, s in text_nbrs:
            print(f"  {clip_df2.iloc[j]['screen_key']}  sim={s:.4f}")

        # similarity fusion neighbors: compute sims on the fly in blocks
        # sim_fused = alpha*(clip@q) + (1-alpha)*(text@q)
        # We'll approximate by computing top-k from fused_vec directly (often close enough)
        fused_nbrs = topk_cosine_neighbors(fused_vec, qi, k=5)
        print("FUSED (concat) neighbors:")
        for j, s in fused_nbrs:
            print(f"  {clip_df2.iloc[j]['screen_key']}  sim={s:.4f}")

    # ---- clustering (use MiniBatchKMeans for scale) ----
    print("\n=== Clustering (MiniBatchKMeans baseline) ===")
    k = 80  # start with 50–200; 80 is a decent first probe
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=10)
    clusters = km.fit_predict(fused_vec)

    out = clip_df2[["screen_key", "app_id", "trace_id", "screen_id"]].copy()
    out["cluster_id"] = clusters
    out.to_csv("screen_clusters.tsv", sep="\t", index=False)

    counts = out["cluster_id"].value_counts()
    print(f"Saved: screen_clusters.tsv")
    print(f"Cluster sizes: min={counts.min()} median={int(counts.median())} max={counts.max()} num_clusters={len(counts)}")


if __name__ == "__main__":
    main()


"""
Alignment + fusion + clustering pipeline (clean and minimal)
What we will do

Load TSVs

Build screen_key

Intersect

Load aligned vectors from .npy

Normalize

Validate fusion by neighbor inspection

Cluster (k-means first, HDBSCAN later)


"""