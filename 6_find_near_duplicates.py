import numpy as np
import pandas as pd
from tqdm import tqdm

def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

def load_clip(clip_tsv, clip_npy):
    df = pd.read_csv(clip_tsv, sep="\t", low_memory=False)
    X = np.load(clip_npy).astype(np.float32)
    assert len(df) == X.shape[0]
    # ensure keys exist
    if "screen_key" not in df.columns:
        if "screen_id" not in df.columns:
            df["screen_id"] = df["image_path"].apply(lambda p: str(p).split("\\")[-1].split("/")[-1].rsplit(".", 1)[0])
        df["screen_key"] = df["app_id"].astype(str) + "::" + df["trace_id"].astype(str) + "::" + df["screen_id"].astype(str)
    return df, X

def find_top1_similarities(X, app_ids, block_q=1024, block_db=8192):
    """
    For each i, compute max cosine similarity to any j!=i.
    X must be L2-normalized.
    Returns:
      best_sim (N,), best_j (N,)
    """
    N, D = X.shape
    best_sim = np.full(N, -np.inf, dtype=np.float32)
    best_j = np.full(N, -1, dtype=np.int32)

    for qs in tqdm(range(0, N, block_q), desc="Query blocks"):
        qe = min(qs + block_q, N)
        Q = X[qs:qe]  # (bq, D)

        # track best for this block locally
        block_best_sim = np.full(qe-qs, -np.inf, dtype=np.float32)
        block_best_j = np.full(qe-qs, -1, dtype=np.int32)

        for ds in range(0, N, block_db):
            de = min(ds + block_db, N)
            B = X[ds:de]  # (bd, D)
            sims = Q @ B.T  # (bq, bd)

            # assume app_ids is an ndarray aligned with rows of X
            # build boolean mask of same-app matches for this tile
            same_app = (app_ids[qs:qe][:, None] == app_ids[ds:de][None, :])

            # exclude self matches (diagonal inside overlap) and same-app matches
            if ds <= qs < de or ds < qe <= de or (qs <= ds and de <= qe):
                # create diagonal mask for overlap region
                idx_q = np.arange(qs, qe)[:, None] - qs
                idx_db = np.arange(ds, de)[None, :] - ds
                diag_mask = (idx_q == idx_db)
                sims[diag_mask] = -np.inf

            sims[same_app] = -np.inf

            # local maxima within this db block
            j_local = np.argmax(sims, axis=1)
            s_local = sims[np.arange(qe-qs), j_local]

            # update block best
            better = s_local > block_best_sim
            block_best_sim[better] = s_local[better]
            block_best_j[better] = (ds + j_local[better]).astype(np.int32)

        best_sim[qs:qe] = block_best_sim
        best_j[qs:qe] = block_best_j

    return best_sim, best_j

def main(
    clip_tsv="processed_data/clip/clip_embeddings_with_ids.tsv",
    clip_npy="processed_data/clip/clip_embeddings.npy",
    out_tsv="near_duplicates_top1_different_apps.tsv",
    threshold=0.99995,
    block_q=1024,
    block_db=8192
):
    df, X = load_clip(clip_tsv, clip_npy)
    X = l2_normalize(X)

    #best_sim, best_j = find_top1_similarities(X, block_q=block_q, block_db=block_db)

    best_sim, best_j = find_top1_similarities(X, app_ids=df["app_id"].values, block_q=block_q, block_db=block_db)
    
    df_out = pd.DataFrame({
        "screen_key": df["screen_key"].values,
        "top1_neighbor_key": df["screen_key"].values[best_j],
        "top1_sim": best_sim,
        "app_id": df["app_id"].values,
        "trace_id": df["trace_id"].values,
        "screen_id": df["screen_id"].values,
        "image_path": df.get("image_path", pd.Series([""]*len(df))).values,
    })

    hits = df_out[df_out["top1_sim"] >= threshold].copy()
    hits.sort_values("top1_sim", ascending=False, inplace=True)

    hits.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved: {out_tsv}")
    print(f"Threshold: {threshold}")
    print(f"Count >= threshold: {len(hits)} / {len(df_out)}")

if __name__ == "__main__":
    main()
