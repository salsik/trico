# make_fused_embeddings.py
# Create fused_embeddings.npy aligned with screen_clusters.tsv order.
# Assumes:
#  - clip_embeddings_with_ids.tsv + clip_embeddings.npy exist
#  - sbert_text_meta.tsv (or similar) + sbert_text_embeddings.npy exist
#  - screen_clusters.tsv contains screen_key (and typically app_id/trace_id/screen_id/cluster_id)
#
# Output:
#  - fused_embeddings.npy  (N x (768+384)=1152) aligned with screen_clusters.tsv row order
from __future__ import annotations
import os
import numpy as np
import pandas as pd


from typing import Tuple

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

def ensure_clip_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "screen_id" not in df.columns:
        if "image_path" not in df.columns:
            raise KeyError("CLIP TSV missing screen_id and image_path; cannot derive screen_id.")
        df["screen_id"] = df["image_path"].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    if "screen_key" not in df.columns:
        df["screen_key"] = df["app_id"].astype(str) + "::" + df["trace_id"].astype(str) + "::" + df["screen_id"].astype(str)
    return df

def ensure_text_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "screen_key" in df.columns:
        return df
    # build from columns if present
    needed = {"app_id", "trace_id", "screen_id"}
    if needed.issubset(set(df.columns)):
        df["screen_key"] = df["app_id"].astype(str) + "::" + df["trace_id"].astype(str) + "::" + df["screen_id"].astype(str)
        return df
    raise KeyError("TEXT meta TSV must have screen_key or (app_id, trace_id, screen_id).")

def load_clip(clip_tsv: str, clip_npy: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(clip_tsv, sep="\t", low_memory=False)
    df = ensure_clip_keys(df)
    X = np.load(clip_npy).astype(np.float32)
    if len(df) != X.shape[0]:
        raise ValueError(f"CLIP mismatch: TSV rows={len(df)} vs NPY rows={X.shape[0]}")
    return df[["screen_key"]], X

def load_text(text_meta_tsv: str, text_npy: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(text_meta_tsv, sep="\t", low_memory=False)
    df = ensure_text_keys(df)
    X = np.load(text_npy).astype(np.float32)
    if len(df) != X.shape[0]:
        raise ValueError(f"TEXT mismatch: TSV rows={len(df)} vs NPY rows={X.shape[0]}")
    return df[["screen_key"]], X

def main(
        
    screen_clusters_tsv: str = "processed_data/clusters/screen_clusters.tsv",
    clip_tsv: str = "processed_data/clip/clip_embeddings_with_ids.tsv",
    clip_npy: str = "processed_data/clip/clip_embeddings.npy",
    text_meta_tsv: str = "processed_data/sbert/sbert_text_meta.tsv",
    text_npy: str = "processed_data/sbert/sbert_text_embeddings.npy",
    out_npy: str = "processed_data/fused/fused_embeddings.npy",
    alpha: float = 1.0,
    beta: float = 1.0,
):
    clusters = pd.read_csv(screen_clusters_tsv, sep="\t", low_memory=False)
    if "screen_key" not in clusters.columns:
        raise KeyError("screen_clusters.tsv must contain screen_key column.")
    target_keys = clusters["screen_key"].astype(str).tolist()
    target_index = {k: i for i, k in enumerate(target_keys)}
    N = len(target_keys)

    clip_df, clip_X = load_clip(clip_tsv, clip_npy)
    text_df, text_X = load_text(text_meta_tsv, text_npy)

    # Build dict key -> row index for quick lookup
    clip_map = {k: i for i, k in enumerate(clip_df["screen_key"].astype(str).tolist())}
    text_map = {k: i for i, k in enumerate(text_df["screen_key"].astype(str).tolist())}

    # Ensure all target keys exist (or warn)
    missing_clip = [k for k in target_keys if k not in clip_map]
    missing_text = [k for k in target_keys if k not in text_map]
    if missing_clip or missing_text:
        raise ValueError(
            f"Missing keys: clip={len(missing_clip)}, text={len(missing_text)}. "
            "Ensure screen_clusters.tsv was generated from the aligned intersection."
        )

    # Gather in exactly the same order as screen_clusters.tsv
    clip_idx = np.array([clip_map[k] for k in target_keys], dtype=np.int64)
    text_idx = np.array([text_map[k] for k in target_keys], dtype=np.int64)

    clip_X2 = clip_X[clip_idx]
    text_X2 = text_X[text_idx]

    # Normalize each modality then fuse
    clip_X2 = l2_normalize(clip_X2)
    text_X2 = l2_normalize(text_X2)

    fused = np.concatenate([alpha * clip_X2, beta * text_X2], axis=1).astype(np.float32)
    fused = l2_normalize(fused)

    np.save(out_npy, fused)
    print(f"Saved: {out_npy}  shape={fused.shape}  (aligned with {screen_clusters_tsv})")

if __name__ == "__main__":
    main()



## 3_python make_fused_embeddings.py
