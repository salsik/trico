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

def get_num_clusters(seqs):
    mx = -1
    for _, _, seq in seqs:
        if seq:
            mx = max(mx, max(seq))
    return mx + 1

def recall_from_ranked(true_next, ranked, k):
    return 1 if true_next in ranked[:k] else 0

# ----- models -----
def train_unigram_next(train_seqs, K, alpha=1.0):
    counts = np.zeros(K, dtype=np.float64)
    for _, _, seq in train_seqs:
        for _, b in zip(seq[:-1], seq[1:]):
            counts[b] += 1.0
    counts += alpha
    probs = counts / counts.sum()
    return probs

def train_1gram(train_seqs, K, alpha=1.0):
    counts = np.zeros((K, K), dtype=np.float64)
    for _, _, seq in train_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1.0
    counts += alpha
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs

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

# ----- evaluation -----
def eval_unigram_trigram_subset(test_seqs, probs, k_list=(1,3,5,10)):
    hits = {k: 0 for k in k_list}
    total = 0
    ranked = np.argsort(-probs)
    for _, _, seq in test_seqs:
        if len(seq) < 3:
            continue
        for _, _, n in zip(seq[:-2], seq[1:-1], seq[2:]):
            total += 1
            for k in k_list:
                hits[k] += recall_from_ranked(n, ranked, k)
    return {k: hits[k]/max(1,total) for k in k_list}, total

def eval_1gram_trigram_subset(test_seqs, probs, k_list=(1,3,5,10)):
    hits = {k: 0 for k in k_list}
    total = 0
    for _, _, seq in test_seqs:
        if len(seq) < 3:
            continue
        for _, c, n in zip(seq[:-2], seq[1:-1], seq[2:]):
            total += 1
            ranked = np.argsort(-probs[c])
            for k in k_list:
                hits[k] += recall_from_ranked(n, ranked, k)
    return {k: hits[k]/max(1,total) for k in k_list}, total

def eval_2gram_trigram_subset(test_seqs, tri_counts, bi_probs, k_list=(1,3,5,10), lambda_backoff=0.2):
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
    return {k: hits[k]/max(1,total) for k in k_list}, total

def main(
    clusters_tsv="processed_data/clusters/screen_clusters_k40.tsv",
    app_category_tsv="../../dataset/6- app_details.csv",
    lambda_backoff=0.5,
    alpha_smooth=1.0,
    min_apps_per_category=50
):
    k_list = (1,3,5,10)

    seqs = build_sequences(clusters_tsv)
    K = get_num_clusters(seqs)

    with open(app_category_tsv, "r", encoding="utf-8", errors="replace") as f:
        cat_df = pd.read_csv(f, low_memory=False)

    #cat_df = pd.read_csv(app_category_tsv,encoding="cp1252", low_memory=False)
    cat_map = dict(zip(cat_df["App_Package_Name"].astype(str), cat_df["Category"].astype(str)))

    # attach categories, drop apps with unknown category
    seqs2 = []
    for app, trace, seq in seqs:
        cat = cat_map.get(str(app), None)
        if cat is not None:
            seqs2.append((app, trace, seq, cat))

    # group apps by category
    apps_by_cat = defaultdict(set)
    for app, _, _, cat in seqs2:
        apps_by_cat[cat].add(app)

    cats = [c for c in sorted(apps_by_cat.keys()) if len(apps_by_cat[c]) >= min_apps_per_category]
    print(f"K={K} | total traces={len(seqs2)} | categories used={len(cats)} (min_apps_per_category={min_apps_per_category})")

    rows = []
    for heldout in cats:
        train = [(a,t,s) for (a,t,s,c) in seqs2 if c != heldout]
        test  = [(a,t,s) for (a,t,s,c) in seqs2 if c == heldout]

        uni = train_unigram_next(train, K, alpha=alpha_smooth)
        p1  = train_1gram(train, K, alpha=alpha_smooth)
        tri_counts, bi_counts = train_2gram(train, K)
        bi_probs = bi_counts + alpha_smooth
        bi_probs = bi_probs / bi_probs.sum(axis=1, keepdims=True)

        uni_sc, e_uni = eval_unigram_trigram_subset(test, uni, k_list=k_list)
        p1_sc,  e_1   = eval_1gram_trigram_subset(test, p1,  k_list=k_list)
        p2_sc,  e_2   = eval_2gram_trigram_subset(test, tri_counts, bi_probs, k_list=k_list, lambda_backoff=lambda_backoff)

        rows.append({
            "heldout_category": heldout,
            "test_apps": len(apps_by_cat[heldout]),
            "edges_eval": e_2,
            "uni_R1": uni_sc[1], "uni_R3": uni_sc[3], "uni_R5": uni_sc[5], "uni_R10": uni_sc[10],
            "1g_R1": p1_sc[1], "1g_R3": p1_sc[3], "1g_R5": p1_sc[5], "1g_R10": p1_sc[10],
            "2g_R1": p2_sc[1], "2g_R3": p2_sc[3], "2g_R5": p2_sc[5], "2g_R10": p2_sc[10],
        })

        print(f"\nHeld-out: {heldout} | test_apps={len(apps_by_cat[heldout])} | edges={e_2}")
        print(f"  Unigram  R@1={uni_sc[1]:.4f} R@3={uni_sc[3]:.4f} R@5={uni_sc[5]:.4f} R@10={uni_sc[10]:.4f}")
        print(f"  1-gram   R@1={p1_sc[1]:.4f} R@3={p1_sc[3]:.4f} R@5={p1_sc[5]:.4f} R@10={p1_sc[10]:.4f}")
        print(f"  2-gram   R@1={p2_sc[1]:.4f} R@3={p2_sc[3]:.4f} R@5={p2_sc[5]:.4f} R@10={p2_sc[10]:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv("category_leave_one_out_results.tsv", sep="\t", index=False)
    print("\nSaved: category_leave_one_out_results.tsv")

    # macro averages across categories
    if len(out) > 0:
        print("\n=== Macro avg over held-out categories ===")
        for prefix in ["uni", "1g", "2g"]:
            r1  = out[f"{prefix}_R1"].mean()
            r3  = out[f"{prefix}_R3"].mean()
            r5  = out[f"{prefix}_R5"].mean()
            r10 = out[f"{prefix}_R10"].mean()
            print(f"{prefix:>4s}  R@1={r1:.4f} R@3={r3:.4f} R@5={r5:.4f} R@10={r10:.4f}")

if __name__ == "__main__":
    
    main()



### python 11_category_leave_one_out.py