import os
import math
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def parse_screen_id_for_sort(s):
    try:
        return int(str(s).split("_")[0])
    except:
        return str(s)

def build_sequences_from_clusters(clusters_df):
    """
    Returns list of tuples:
      (app_id, trace_id, [cluster_id...], [screen_id...]) aligned by time order
    Consecutive duplicate clusters are collapsed, keeping the first screen_id for that run.
    """
    seqs = []
    for (app, trace), g in clusters_df.groupby(["app_id", "trace_id"]):
        g2 = g.copy()
        g2["sort_key"] = g2["screen_id"].apply(parse_screen_id_for_sort)
        g2 = g2.sort_values("sort_key")

        clusters = g2["cluster_id"].astype(int).tolist()
        screens  = g2["screen_id"].astype(str).tolist()

        if len(clusters) < 2:
            continue

        # collapse consecutive duplicates (cluster-wise), keep first screen_id in each run
        c_comp = [clusters[0]]
        s_comp = [screens[0]]
        for c, sid in zip(clusters[1:], screens[1:]):
            if c != c_comp[-1]:
                c_comp.append(c)
                s_comp.append(sid)

        if len(c_comp) >= 2:
            seqs.append((str(app), str(trace), c_comp, s_comp))
    return seqs

def split_by_app(seqs, test_ratio=0.2, seed=42):
    apps = sorted(list({a for a, _, _, _ in seqs}))
    rng = np.random.default_rng(seed)
    rng.shuffle(apps)
    n_test = max(1, int(len(apps) * test_ratio))
    test_apps = set(apps[:n_test])
    train = [x for x in seqs if x[0] not in test_apps]
    test  = [x for x in seqs if x[0] in test_apps]
    return train, test, test_apps

def train_bigram(train_seqs, K, alpha=1.0):
    counts = np.zeros((K, K), dtype=np.float64)
    for _, _, cseq, _ in train_seqs:
        for a, b in zip(cseq[:-1], cseq[1:]):
            counts[a, b] += 1.0
    counts += alpha
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs

def train_trigram_counts(train_seqs):
    tri_counts = defaultdict(Counter)  # (p,c)->Counter(next)
    for _, _, cseq, _ in train_seqs:
        if len(cseq) < 3:
            continue
        for p, c, n in zip(cseq[:-2], cseq[1:-1], cseq[2:]):
            tri_counts[(p, c)][n] += 1
    return tri_counts

def score_next_prob(p, c, n, tri_counts, bi_probs, lambda_backoff=0.5):
    """
    Probability under mixture:
      (1-lambda)*P_tri(n|p,c) + lambda*P_bi(n|c)
    If trigram unseen, fall back to P_bi(n|c).
    """
    key = (p, c)
    p_bi = float(bi_probs[c, n])
    if key not in tri_counts or len(tri_counts[key]) == 0:
        return p_bi

    counter = tri_counts[key]
    total = sum(counter.values())
    p_tri = (counter[n] / total) if total > 0 else 0.0
    return (1.0 - lambda_backoff) * p_tri + lambda_backoff * p_bi

def topk_predictions(p, c, tri_counts, bi_probs, k=5, lambda_backoff=0.5):
    key = (p, c)
    # Candidate pool: top trigram + top bigram
    if key in tri_counts and len(tri_counts[key]) > 0:
        counter = tri_counts[key]
        tri_top = [x for x, _ in counter.most_common(max(50, k * 10))]
    else:
        tri_top = []

    bi_top = np.argsort(-bi_probs[c])[:max(50, k * 10)].tolist()
    cand = list(dict.fromkeys(tri_top + bi_top))

    scores = []
    if key in tri_counts and len(tri_counts[key]) > 0:
        counter = tri_counts[key]
        total = sum(counter.values())
        for n in cand:
            p_tri = (counter[n] / total) if total > 0 else 0.0
            p_bi = float(bi_probs[c, n])
            scores.append((1.0 - lambda_backoff) * p_tri + lambda_backoff * p_bi)
    else:
        # purely bigram
        for n in cand:
            scores.append(float(bi_probs[c, n]))

    idx = np.argsort(-np.array(scores))[:k]
    return [cand[i] for i in idx], [scores[i] for i in idx]

def safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src is None or (not isinstance(src, str)) or (not os.path.exists(src)):
        # create a placeholder text file so you see missing paths clearly
        with open(dst + ".MISSING.txt", "w", encoding="utf-8") as f:
            f.write(f"Missing source file: {src}\n")
        return False
    shutil.copy2(src, dst)
    return True

def main(
    clusters_tsv,
    screen_to_image_tsv,
    out_dir="anomaly_out",
    test_ratio=0.2,
    seed=42,
    alpha_smooth=1.0,
    lambda_backoff=0.5,
    topN=50,
    topK_preds=5
):
    os.makedirs(out_dir, exist_ok=True)

    clusters_df = pd.read_csv(clusters_tsv, sep="\t")
    required_cols = ["app_id", "trace_id", "screen_id", "cluster_id"]
    for c in required_cols:
        if c not in clusters_df.columns:
            raise ValueError(f"{clusters_tsv} missing required column: {c}")

    map_df = pd.read_csv(screen_to_image_tsv, sep="\t")
    # expect these columns:
    map_required = ["app_id", "trace_id", "screen_id", "image_path"]
    for c in map_required:
        if c not in map_df.columns:
            raise ValueError(f"{screen_to_image_tsv} missing required column: {c}")

    # Build mapping: screen_key -> image_path
    map_df["screen_key"] = map_df["app_id"].astype(str) + "::" + map_df["trace_id"].astype(str) + "::" + map_df["screen_id"].astype(str)
    key_to_img = dict(zip(map_df["screen_key"], map_df["image_path"]))

    # Build sequences
    seqs = build_sequences_from_clusters(clusters_df)
    print(f"Total traces (len>=2, collapsed): {len(seqs)}")

    # K inferred from cluster IDs
    K = int(clusters_df["cluster_id"].max()) + 1
    print(f"K={K}")

    train, test, test_apps = split_by_app(seqs, test_ratio=test_ratio, seed=seed)
    print(f"App split: train_traces={len(train)} test_traces={len(test)} test_apps={len(test_apps)}")

    # Train bigram + trigram
    bi_probs = train_bigram(train, K, alpha=alpha_smooth)
    tri_counts = train_trigram_counts(train)

    # Build representative screen per cluster (from TRAIN only)
    rep_img_by_cluster = {}
    for app, trace, cseq, sseq in train:
        for c, sid in zip(cseq, sseq):
            if c in rep_img_by_cluster:
                continue
            screen_key = f"{app}::{trace}::{sid}"
            rep_img_by_cluster[c] = key_to_img.get(screen_key)

    # Score anomalies on TEST trigram edges
    rows = []
    for app, trace, cseq, sseq in test:
        if len(cseq) < 3:
            continue
        for i in range(2, len(cseq)):
            p, c, n = cseq[i-2], cseq[i-1], cseq[i]
            sid_p, sid_c, sid_n = sseq[i-2], sseq[i-1], sseq[i]

            prob = score_next_prob(p, c, n, tri_counts, bi_probs, lambda_backoff=lambda_backoff)
            prob = max(prob, 1e-12)
            anomaly = -math.log(prob)

            screen_key_p = f"{app}::{trace}::{sid_p}"
            screen_key_c = f"{app}::{trace}::{sid_c}"
            screen_key_n = f"{app}::{trace}::{sid_n}"

            rows.append({
                "app_id": app,
                "trace_id": trace,
                "idx": i,
                "prev_cluster": p,
                "curr_cluster": c,
                "next_cluster": n,
                "prev_screen_id": sid_p,
                "curr_screen_id": sid_c,
                "next_screen_id": sid_n,
                "prev_screen_key": screen_key_p,
                "curr_screen_key": screen_key_c,
                "next_screen_key": screen_key_n,
                "prev_image": key_to_img.get(screen_key_p),
                "curr_image": key_to_img.get(screen_key_c),
                "next_image": key_to_img.get(screen_key_n),
                "next_prob": prob,
                "anomaly": anomaly
            })

    if not rows:
        raise RuntimeError("No trigram edges found in test set. Check your traces after collapsing duplicates.")

    df = pd.DataFrame(rows).sort_values("anomaly", ascending=False)
    out_tsv = os.path.join(out_dir, "top_anomalies.tsv")
    df.head(topN).to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved: {out_tsv}")

    # Export images for topN anomalies
    export_root = os.path.join(out_dir, "cases_top_anomalies")
    os.makedirs(export_root, exist_ok=True)

    for rank, r in enumerate(df.head(topN).itertuples(index=False), start=1):
        case_dir = os.path.join(export_root, f"{rank:03d}_{r.app_id}_{r.trace_id}_i{r.idx}")
        os.makedirs(case_dir, exist_ok=True)

        # Copy prev/curr/next
        safe_copy(r.prev_image, os.path.join(case_dir, "prev.png"))
        safe_copy(r.curr_image, os.path.join(case_dir, "curr.png"))
        safe_copy(r.next_image, os.path.join(case_dir, "next.png"))

        # Top-K predictions + representative screenshots
        preds, pred_scores = topk_predictions(
            r.prev_cluster, r.curr_cluster,
            tri_counts, bi_probs,
            k=topK_preds,
            lambda_backoff=lambda_backoff
        )

        with open(os.path.join(case_dir, "predictions.txt"), "w", encoding="utf-8") as f:
            f.write(f"anomaly={r.anomaly:.6f}  next_prob={r.next_prob:.8f}\n")
            f.write(f"context: ({r.prev_cluster} -> {r.curr_cluster})  true_next={r.next_cluster}\n\n")
            for j, (cl, sc) in enumerate(zip(preds, pred_scores), start=1):
                f.write(f"top{j}: cluster={cl}  score={sc:.8f}\n")

        # Export representative images for predicted clusters
        for j, cl in enumerate(preds, start=1):
            rep = rep_img_by_cluster.get(cl, None)
            safe_copy(rep, os.path.join(case_dir, f"pred_top{j}_cluster{cl}.png"))

    print(f"Exported cases to: {export_root}")
    print("\nTip: open any case folder and compare prev/curr/next + predicted cluster exemplars.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analysis/13_mine_anomalies_and_export.py <clusters.tsv> <screen_to_image.tsv>")
        print("Example: python analysis/13_mine_anomalies_and_export.py processed_data/clusters/screen_clusters_k40.tsv processed_data/clip/clip_embeddings_with_ids.tsv")
        sys.exit(1)

    main(
        clusters_tsv=sys.argv[1],
        screen_to_image_tsv=sys.argv[2],
        out_dir="anomaly_out",
        test_ratio=0.2,
        seed=42,
        alpha_smooth=1.0,
        lambda_backoff=0.5,   # use your best lambda (0.5 looked best)
        topN=50,
        topK_preds=5
    )



## python 13_mine_anomalies_and_export.py processed_data/clusters/screen_clusters_k40.tsv processed_data/clip/clip_embeddings_with_ids.tsv
