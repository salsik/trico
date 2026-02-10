import pandas as pd
import numpy as np

def main(
    clusters_tsv="processed_data/clusters/screen_clusters.tsv",
    clip_tsv="processed_data/clip/clip_embeddings_with_ids.tsv",
    out_dir="cluster_samples",
    samples_per_cluster=20,
    seed=42
):
    import os
    os.makedirs(out_dir, exist_ok=True)

    clusters = pd.read_csv(clusters_tsv, sep="\t")
    clip = pd.read_csv(clip_tsv, sep="\t", low_memory=False)

    # Ensure screen_key exists in both
    if "screen_key" not in clip.columns:
        if "screen_id" not in clip.columns:
            clip["screen_id"] = clip["image_path"].apply(lambda p: str(p).split("\\")[-1].split("/")[-1].rsplit(".", 1)[0])
        clip["screen_key"] = clip["app_id"].astype(str) + "::" + clip["trace_id"].astype(str) + "::" + clip["screen_id"].astype(str)

    merged = clusters.merge(clip[["screen_key", "image_path"]], on="screen_key", how="left")

    rng = np.random.default_rng(seed)
    for cid, g in merged.groupby("cluster_id"):
        if len(g) == 0:
            continue
        take = min(samples_per_cluster, len(g))
        idx = rng.choice(len(g), size=take, replace=False)
        sample = g.iloc[idx].copy()
        sample.to_csv(f"{out_dir}/cluster_{cid}_samples.tsv", sep="\t", index=False)

    # also export largest clusters list
    sizes = merged["cluster_id"].value_counts().reset_index()
    sizes.columns = ["cluster_id", "size"]
    sizes.to_csv(f"{out_dir}/cluster_sizes.tsv", sep="\t", index=False)

    print(f"Saved samples to {out_dir}/")

if __name__ == "__main__":
    main()
