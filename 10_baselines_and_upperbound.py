import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def parse_screen_id_for_sort(s):
    try:
        return int(str(s).split("_")[0])
    except:
        return str(s)

def build_sequences(clusters_tsv):
    df = pd.read_csv(clusters_tsv, sep="\t")
    seqs = []
    for (app, trace), g in df.groupby(["app_id", "trace_id"]):
        g2 = g.copy()
        g2["sort_key"] = g2["screen_id"].apply(parse_screen_id_for_sort)
        g2 = g2.sort_values("sort_key")
        seq = g2["cluster_id"].astype(int).tolist()

        # collapse consecutive duplicates
        if len(seq) >= 2:
            comp = [seq[0]]
            for c in seq[1:]:
                if c != comp[-1]:
                    comp.append(c)
            if len(comp) >= 2:
                seqs.append((app, trace, comp))
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

def get_num_clusters(seqs):
    mx = -1
    for _, _, seq in seqs:
        if seq:
            mx = max(mx, max(seq))
    return mx + 1

# ---------- helpers ----------
def recall_from_ranked(true_next, ranked, k):
    return 1 if true_next in ranked[:k] else 0

# ---------- Unigram baseline (global next cluster frequency) ----------
def train_unigram_next(train_seqs, K, alpha=1.0):
    # counts of next states across all transitions
    counts = np.zeros(K, dtype=np.float64)
    for _, _, seq in train_seqs:
        for _, b in zip(seq[:-1], seq[1:]):
            counts[b] += 1.0
    counts += alpha
    probs = counts / counts.sum()
    return probs

def eval_unigram(test_seqs, probs, k_list=(1,3,5,10), trigram_subset=False):
    # if trigram_subset=True, evaluate only positions with history (len>=3)
    hits = {k: 0 for k in k_list}
    total = 0
    ranked = np.argsort(-probs)  # global ranking

    for _, _, seq in test_seqs:
        if trigram_subset and len(seq) < 3:
            continue
        if trigram_subset:
            triples = zip(seq[:-2], seq[1:-1], seq[2:])
            for _, _, n in triples:
                total += 1
                for k in k_list:
                    hits[k] += recall_from_ranked(n, ranked, k)
        else:
            for _, n in zip(seq[:-1], seq[1:]):
                total += 1
                for k in k_list:
                    hits[k] += recall_from_ranked(n, ranked, k)

    return {k: hits[k] / max(1, total) for k in k_list}, total

# ---------- Cross-app 1-gram ----------
def train_1gram(train_seqs, K, alpha=1.0):
    counts = np.zeros((K, K), dtype=np.float64)
    for _, _, seq in train_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1.0
    counts += alpha
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs

def eval_1gram(test_seqs, probs, k_list=(1,3,5,10), trigram_subset=False):
    hits = {k: 0 for k in k_list}
    total = 0

    for _, _, seq in test_seqs:
        if trigram_subset and len(seq) < 3:
            continue
        if trigram_subset:
            triples = zip(seq[:-2], seq[1:-1], seq[2:])
            for _, c, n in triples:
                total += 1
                ranked = np.argsort(-probs[c])
                for k in k_list:
                    hits[k] += recall_from_ranked(n, ranked, k)
        else:
            for c, n in zip(seq[:-1], seq[1:]):
                total += 1
                ranked = np.argsort(-probs[c])
                for k in k_list:
                    hits[k] += recall_from_ranked(n, ranked, k)

    return {k: hits[k] / max(1, total) for k in k_list}, total

# ---------- Cross-app 2-gram: P(next | prev, curr) with backoff ----------
def train_2gram(train_seqs, K):
    tri_counts = defaultdict(Counter)   # (p,c)->Counter(next)
    bi_counts = np.zeros((K, K), dtype=np.float64)  # c->next
    for _, _, seq in train_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            bi_counts[a, b] += 1.0
        for p, c, n in zip(seq[:-2], seq[1:-1], seq[2:]):
            tri_counts[(p, c)][n] += 1
    return tri_counts, bi_counts

def predict_topk_2gram(p, c, tri_counts, bi_probs, k, lambda_backoff=0.2):
    key = (p, c)
    if key not in tri_counts or len(tri_counts[key]) == 0:
        return np.argsort(-bi_probs[c])[:k]

    counter = tri_counts[key]
    total = sum(counter.values())

    tri_top = [x for x, _ in counter.most_common(max(k*5, 50))]
    bi_top = np.argsort(-bi_probs[c])[:max(k*5, 50)].tolist()
    cand = list(dict.fromkeys(tri_top + bi_top))

    scores = []
    for n in cand:
        p_tri = counter[n] / total
        p_bi = bi_probs[c, n]
        scores.append((1.0 - lambda_backoff) * p_tri + lambda_backoff * p_bi)

    idx = np.argsort(-np.array(scores))[:k]
    return np.array([cand[i] for i in idx], dtype=np.int64)

def eval_2gram(test_seqs, tri_counts, bi_probs, k_list=(1,3,5,10), lambda_backoff=0.2):
    hits = {k: 0 for k in k_list}
    total = 0
    for _, _, seq in test_seqs:
        if len(seq) < 3:
            continue
        for p, c, n in zip(seq[:-2], seq[1:-1], seq[2:]):
            total += 1
            for k in k_list:
                topk = predict_topk_2gram(p, c, tri_counts, bi_probs, k, lambda_backoff=lambda_backoff)
                hits[k] += (1 if n in topk else 0)
    return {k: hits[k] / max(1, total) for k in k_list}, total

# ---------- App-specific bigram upper bound ----------
def app_specific_bigram_upperbound(seqs, K, alpha=1.0, test_ratio_within_app=0.2, seed=42, k_list=(1,3,5,10)):
    """
    For each app independently:
      - split that app's traces into train/test
      - train bigram P(next|curr) on that app only
      - evaluate on that app's test traces
    Aggregate edges across apps (micro-average).
    """
    rng = np.random.default_rng(seed)
    by_app = defaultdict(list)
    for app, trace, seq in seqs:
        by_app[app].append((trace, seq))

    hits = {k: 0 for k in k_list}
    total = 0

    for app, items in by_app.items():
        if len(items) < 2:
            continue
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_test = max(1, int(len(items) * test_ratio_within_app))
        test_idx = set(idx[:n_test])
        train_items = [items[i] for i in range(len(items)) if i not in test_idx]
        test_items  = [items[i] for i in range(len(items)) if i in test_idx]

        # train bigram for this app
        probs = train_1gram([(app, tr, s) for tr, s in train_items], K, alpha=alpha)

        # eval (on all bigram edges)
        for tr, seq in test_items:
            for c, n in zip(seq[:-1], seq[1:]):
                total += 1
                ranked = np.argsort(-probs[c])
                for k in k_list:
                    hits[k] += recall_from_ranked(n, ranked, k)

    return {k: hits[k] / max(1, total) for k in k_list}, total

def format_row(name, scores):
    return f"{name:28s}  " + "  ".join([f"{scores[k]:.4f}" for k in [1,3,5,10]])

def main(
    clusters_tsv,
    lambda_backoff=0.2,
    alpha_smooth=1.0,
    seed=42,
    test_ratio_apps=0.2
):
    k_list = (1,3,5,10)
    seqs = build_sequences(clusters_tsv)
    K = get_num_clusters(seqs)

    train, test, test_apps = split_by_app(seqs, test_ratio=test_ratio_apps, seed=seed)

    # Unigram
    uni = train_unigram_next(train, K, alpha=alpha_smooth)
    uni_all, edges_uni_all = eval_unigram(test, uni, k_list=k_list, trigram_subset=False)
    uni_sub, edges_uni_sub = eval_unigram(test, uni, k_list=k_list, trigram_subset=True)

    # Cross-app 1-gram
    probs1 = train_1gram(train, K, alpha=alpha_smooth)
    one_all, edges1_all = eval_1gram(test, probs1, k_list=k_list, trigram_subset=False)
    one_sub, edges1_sub = eval_1gram(test, probs1, k_list=k_list, trigram_subset=True)

    # Cross-app 2-gram
    tri_counts, bi_counts = train_2gram(train, K)
    bi_probs = bi_counts + alpha_smooth
    bi_probs = bi_probs / bi_probs.sum(axis=1, keepdims=True)
    two_sub, edges2_sub = eval_2gram(test, tri_counts, bi_probs, k_list=k_list, lambda_backoff=lambda_backoff)

    # App-specific upper bound (bigram)
    # no need for now
    #upper, edges_upper = app_specific_bigram_upperbound(seqs, K, alpha=alpha_smooth, test_ratio_within_app=0.2, seed=seed, k_list=k_list)

    print(f"\nFile: {clusters_tsv}")
    print(f"Traces (len>=2): {len(seqs)} | K={K}")
    print(f"App-split: train_traces={len(train)} test_traces={len(test)} test_apps={len(test_apps)}")
    print(f"lambda_backoff={lambda_backoff} | alpha_smooth={alpha_smooth}")

    print("\nEdges evaluated:")
    print(f"  Unigram all: {edges_uni_all} | Unigram trigram-subset: {edges_uni_sub}")
    print(f"  1-gram all:  {edges1_all} | 1-gram trigram-subset:  {edges1_sub}")
    print(f"  2-gram trigram-subset: {edges2_sub}")
    #print(f"  App-specific bigram (upper bound) edges: {edges_upper}")

    print("\nRecall@k (k=1,3,5,10)")
    print(" " * 30 + "R@1     R@3     R@5     R@10")
    print(format_row("Unigram (global, all)", uni_all))
    print(format_row("Unigram (global, subset)", uni_sub))
    print(format_row("Cross-app 1-gram (all)", one_all))
    print(format_row("Cross-app 1-gram (subset)", one_sub))
    print(format_row("Cross-app 2-gram (subset)", two_sub))
    #rint(format_row("App-specific 1-gram (UB)", upper))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analysis/10_baselines_and_upperbound.py <screen_clusters_kXX.tsv>")
        sys.exit(1)
    main(sys.argv[1])


###

"""

python 10_baselines_and_upperbound.py processed_data/clusters/screen_clusters_k40.tsv
python 10_baselines_and_upperbound.py processed_data/clusters/screen_clusters_k80.tsv
python 10_baselines_and_upperbound.py processed_data/clusters/screen_clusters_k120.tsv
"""