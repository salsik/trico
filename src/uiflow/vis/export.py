from __future__ import annotations
import os
import numpy as np
import pandas as pd

def export_cluster_samples(
    clusters_df: pd.DataFrame,
    image_meta_df: pd.DataFrame,
    out_dir: str,
    samples_per_cluster: int = 20,
    seed: int = 42,
) -> None:
    """
    Save per-cluster sample TSVs containing screen_key and image_path (if available).
    """
    os.makedirs(out_dir, exist_ok=True)

    if "screen_key" not in image_meta_df.columns:
        raise KeyError("image_meta_df must have screen_key")
    if "screen_key" not in clusters_df.columns or "cluster_id" not in clusters_df.columns:
        raise KeyError("clusters_df must have screen_key and cluster_id")

    merged = clusters_df.merge(image_meta_df[["screen_key", "image_path"]], on="screen_key", how="left")

    rng = np.random.default_rng(seed)
    for cid, g in merged.groupby("cluster_id"):
        take = min(samples_per_cluster, len(g))
        idx = rng.choice(len(g), size=take, replace=False)
        sample = g.iloc[idx].copy()
        sample.to_csv(os.path.join(out_dir, f"cluster_{cid}_samples.tsv"), sep="\t", index=False)

    sizes = merged["cluster_id"].value_counts().reset_index()
    sizes.columns = ["cluster_id", "size"]
    sizes.to_csv(os.path.join(out_dir, "cluster_sizes.tsv"), sep="\t", index=False)
