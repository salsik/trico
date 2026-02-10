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

# -------- 1-gram model: P(next | curr) --------
def train_1gram(train_seqs, K, alpha=1.0):
    counts = np.zeros((K, K), dtype=np.float64)
    for _, _, seq in train_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1.0
    counts += alpha
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs

def predict_topk_1gram(probs_row, k):
    return np.argsort(-probs_row)[:k]

def eval_recall_1gram(test_seqs, probs, k_list=(1,3,5,10)):
    hits = {k: 0 for k in k_list}
    total = 0
    for _, _, seq in test_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            total += 1
            row = probs[a]
            for k in k_list:
                if b in predict_topk_1gram(row, k):
                    hits[k] += 1
    return {k: hits[k] / max(1, total) for k in k_list}, total

# -------- 2-gram model: P(next | prev, curr) with backoff --------
def train_2gram(train_seqs, K):
    # trigram counts: (prev, curr) -> next
    tri_counts = defaultdict(Counter)
    # bigram counts for backoff: curr -> next
    bi_counts = np.zeros((K, K), dtype=np.float64)

    for _, _, seq in train_seqs:
        # bigram
        for a, b in zip(seq[:-1], seq[1:]):
            bi_counts[a, b] += 1.0
        # trigram
        for p, c, n in zip(seq[:-2], seq[1:-1], seq[2:]):
            tri_counts[(p, c)][n] += 1

    return tri_counts, bi_counts

def topk_from_counter(counter: Counter, k: int):
    # returns list of (item, count)
    return [x for x, _ in counter.most_common(k)]

def predict_topk_2gram(
    prev_c, curr_c,
    tri_counts,
    bi_probs,     # (K,K) normalized bigram probs
    k: int,
    lambda_backoff: float = 0.2,
    K: int = None
):
    """
    Mixture of:
      - trigram empirical distribution (if exists)
      - backoff bigram distribution P(next|curr)
    If trigram exists, we form a score:
      score(next) = (1-lambda)*P_tri(next|prev,curr) + lambda*P_bi(next|curr)
    If not, fallback to bigram.
    """
    key = (prev_c, curr_c)
    if key not in tri_counts or len(tri_counts[key]) == 0:
        return np.argsort(-bi_probs[curr_c])[:k]

    counter = tri_counts[key]
    total = sum(counter.values())
    # Candidates: union of trigram top candidates and top bigram candidates
    tri_top = [x for x, _ in counter.most_common(max(k*5, 50))]
    bi_top = np.argsort(-bi_probs[curr_c])[:max(k*5, 50)].tolist()
    cand = list(dict.fromkeys(tri_top + bi_top))  # preserve order, unique

    scores = []
    for n in cand:
        p_tri = counter[n] / total
        p_bi = bi_probs[curr_c, n]
        score = (1.0 - lambda_backoff) * p_tri + lambda_backoff * p_bi
        scores.append(score)

    # pick top-k
    idx = np.argsort(-np.array(scores))[:k]
    return np.array([cand[i] for i in idx], dtype=np.int64)

def eval_recall_2gram(test_seqs, tri_counts, bi_probs, k_list=(1,3,5,10), lambda_backoff=0.2):
    hits = {k: 0 for k in k_list}
    total = 0

    for _, _, seq in test_seqs:
        if len(seq) < 3:
            continue
        # predict next using (prev, curr)
        for p, c, n in zip(seq[:-2], seq[1:-1], seq[2:]):
            total += 1
            for k in k_list:
                topk = predict_topk_2gram(p, c, tri_counts, bi_probs, k, lambda_backoff=lambda_backoff)
                if n in topk:
                    hits[k] += 1

    return {k: hits[k] / max(1, total) for k in k_list}, total

def main1(
    clusters_tsv="processed_data/clusters/screen_clusters.tsv",
    test_ratio=0.2,
    seed=42,
    alpha_smooth=1.0,
    lambda_backoff=0.0
):
    seqs = build_sequences(clusters_tsv)
    print(f"Traces with length>=2 (after collapsing dups): {len(seqs)}")
    K = get_num_clusters(seqs)
    print(f"Num clusters: {K}")

    train, test, test_apps = split_by_app(seqs, test_ratio=test_ratio, seed=seed)
    print(f"Train traces: {len(train)} | Test traces: {len(test)} | Test apps: {len(test_apps)}")

    # ---- 1-gram ----
    probs_1 = train_1gram(train, K, alpha=alpha_smooth)
    rec1, edges1 = eval_recall_1gram(test, probs_1, k_list=(1,3,5,10))

    # ---- 2-gram ----
    tri_counts, bi_counts = train_2gram(train, K)
    # normalize bigram probs for backoff
    bi_probs = bi_counts + alpha_smooth
    bi_probs = bi_probs / bi_probs.sum(axis=1, keepdims=True)

    rec2, edges2 = eval_recall_2gram(
        test, tri_counts, bi_probs, k_list=(1,3,5,10), lambda_backoff=lambda_backoff
    )

    print("\n=== Next-step prediction comparison ===")
    print(f"1-gram evaluated edges: {edges1}")
    print(f"2-gram evaluated edges: {edges2}  (requires length>=3)")

    for k in [1,3,5,10]:
        r1 = rec1[k]
        r2 = rec2[k]
        delta = r2 - r1
        print(f"Recall@{k}:  1-gram={r1:.4f}   2-gram={r2:.4f}   delta={delta:+.4f}")

    print(f"\nBackoff lambda (toward bigram): {lambda_backoff}")
    print(f"Additive smoothing alpha: {alpha_smooth}")



def eval_recall_1gram_on_trigram_edges(test_seqs, probs, k_list=(1,3,5,10)):
    """
    Evaluate 1-gram model only on edges where a trigram context exists:
    i.e., positions with (prev, curr) -> next. This matches the 2-gram edge subset.
    """
    hits = {k: 0 for k in k_list}
    total = 0

    for _, _, seq in test_seqs:
        if len(seq) < 3:
            continue
        for p, c, n in zip(seq[:-2], seq[1:-1], seq[2:]):
            total += 1
            row = probs[c]
            # compute top-k once per k by sorting once (small K=80; OK)
            order = np.argsort(-row)
            for k in k_list:
                if n in order[:k]:
                    hits[k] += 1

    return {k: hits[k] / max(1, total) for k in k_list}, total



def main2(
    clusters_tsv="processed_data/clusters/screen_clusters_k120.tsv",
    test_ratio=0.2,
    seed=42,
    alpha_smooth=1.0,
    lambda_backoff=0.2
):
    seqs = build_sequences(clusters_tsv)
    print(f"Traces with length>=2 (after collapsing dups): {len(seqs)}")
    K = get_num_clusters(seqs)
    print(f"Num clusters: {K}")

    train, test, test_apps = split_by_app(seqs, test_ratio=test_ratio, seed=seed)
    print(f"Train traces: {len(train)} | Test traces: {len(test)} | Test apps: {len(test_apps)}")

    # ---- 1-gram ----
    probs_1 = train_1gram(train, K, alpha=alpha_smooth)

    rec1_all, edges1_all = eval_recall_1gram(test, probs_1, k_list=(1,3,5,10))
    rec1_sub, edges1_sub = eval_recall_1gram_on_trigram_edges(test, probs_1, k_list=(1,3,5,10))

    
    # ---- 2-gram ----
    tri_counts, bi_counts = train_2gram(train, K)
    # normalize bigram probs for backoff
    bi_probs = bi_counts + alpha_smooth
    bi_probs = bi_probs / bi_probs.sum(axis=1, keepdims=True)

    rec2, edges2 = eval_recall_2gram(
        test, tri_counts, bi_probs, k_list=(1,3,5,10), lambda_backoff=lambda_backoff
    )


    print("\n=== Next-step prediction comparison ===")
    print(f"1-gram evaluated edges (all bigram edges): {edges1_all}")
    print(f"1-gram evaluated edges (trigram subset):  {edges1_sub}")
    print(f"2-gram evaluated edges (trigram subset):  {edges2}  (requires length>=3)")

    print("\n--- Apple-to-apple (same trigram subset) ---")
    for k in [1,3,5,10]:
        r1 = rec1_sub[k]
        r2 = rec2[k]
        print(f"Recall@{k}:  1-gram={r1:.4f}   2-gram={r2:.4f}   delta={(r2-r1):+.4f}")

    print("\n--- Reference (1-gram on all edges) ---")
    for k in [1,3,5,10]:
        print(f"Recall@{k}:  1-gram(all)={rec1_all[k]:.4f}")

        
    print(f"\nBackoff lambda (toward bigram): {lambda_backoff}")
    print(f"Additive smoothing alpha: {alpha_smooth}")


if __name__ == "__main__":
    ##main1() first case of comparing 2gram with 3gram
    
    # apple-to-apple evaluation
    # Evaluate 1-gram on the same trigram edge set used for 2-gram (i.e., use only transitions where a previous state exists).
    main2() #
