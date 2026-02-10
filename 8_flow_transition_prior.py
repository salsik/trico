import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def parse_screen_id_for_sort(s):
    # screen_id might be "2134" or "2134_something"
    try:
        return int(str(s).split("_")[0])
    except:
        return str(s)

def build_sequences(clusters_tsv):
    df = pd.read_csv(clusters_tsv, sep="\t")
    # group by (app_id, trace_id) and sort by screen_id
    seqs = []
    for (app, trace), g in df.groupby(["app_id", "trace_id"]):
        g2 = g.copy()
        g2["sort_key"] = g2["screen_id"].apply(parse_screen_id_for_sort)
        g2 = g2.sort_values("sort_key")
        seq = g2["cluster_id"].astype(int).tolist()
        # remove consecutive duplicates (optional; helps with identical screens)
        seq_comp = [seq[0]] if seq else []
        for c in seq[1:]:
            if c != seq_comp[-1]:
                seq_comp.append(c)
        if len(seq_comp) >= 2:
            seqs.append((app, trace, seq_comp))
    return seqs

def split_by_app(seqs, test_ratio=0.2, seed=42):
    apps = sorted(list({a for a, _, _ in seqs}))
    rng = np.random.default_rng(seed)
    rng.shuffle(apps)
    n_test = max(1, int(len(apps) * test_ratio))
    test_apps = set(apps[:n_test])

    train = [s for s in seqs if s[0] not in test_apps]
    test = [s for s in seqs if s[0] in test_apps]
    return train, test, test_apps

def train_bigram_model(train_seqs, num_clusters, smoothing=1.0):
    # Count transitions
    counts = np.zeros((num_clusters, num_clusters), dtype=np.float64)
    for _, _, seq in train_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1.0

    # add smoothing
    counts += smoothing
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs

def recall_at_k(test_seqs, probs, k_list=(1,3,5,10)):
    hits = {k: 0 for k in k_list}
    total = 0

    for _, _, seq in test_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            total += 1
            row = probs[a]
            top = np.argsort(-row)  # descending
            for k in k_list:
                if b in top[:k]:
                    hits[k] += 1

    return {k: hits[k] / max(1, total) for k in k_list}, total

def main(
    clusters_tsv="processed_data/clusters/screen_clusters.tsv",
    test_ratio=0.2,
    seed=42,
    smoothing=1.0
):
    seqs = build_sequences(clusters_tsv)
    print(f"Traces with length>=2: {len(seqs)}")

    # Determine cluster count
    all_clusters = set()
    for _, _, seq in seqs:
        all_clusters.update(seq)
    num_clusters = max(all_clusters) + 1
    print(f"Num clusters (from data): {num_clusters}")

    train, test, test_apps = split_by_app(seqs, test_ratio=test_ratio, seed=seed)
    print(f"Train traces: {len(train)} | Test traces: {len(test)} | Test apps: {len(test_apps)}")

    probs = train_bigram_model(train, num_clusters=num_clusters, smoothing=smoothing)
    scores, total_edges = recall_at_k(test, probs, k_list=(1,3,5,10))

    print("\n=== Next-step prediction (cluster bigram) ===")
    print(f"Test edges: {total_edges}")
    for k, v in scores.items():
        print(f"Recall@{k}: {v:.4f}")

    # Save transition matrix (optional)
    np.save("transition_probs.npy", probs.astype(np.float32))
    print("\nSaved: transition_probs.npy")

if __name__ == "__main__":
    main()

"""
Trace → cluster sequences + transition prior + Recall@k (split by app)

This creates the core artifact for your paper.

Assumptions

Within each trace, screen order can be approximated by sorting screen_id numerically (it looks like your filenames are numbers like 2134).

If you have a true ordering file in traces, we can switch later.

python analysis/08_flow_transition_prior.py
What this gives you (paper-grade baseline)
A clean split by app_id (prevents leakage via duplicates)

A simple transition prior

Recall@k numbers you can put in a table immediately

Then we iterate:

increase k-means clusters (80→120→200)

test smoothing

try 2-gram vs 3-gram

add “atypical transition” scoring (-log P)

"""