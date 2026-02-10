# ============================================================
# Notebook script: Export top anomalous REAL test transitions
# using trigram prior anomaly score = -log Pmix(next | prev2, prev1)
#
# What it does:
# 1) Loads TSV (clusters per screen)
# 2) Builds traces (optionally collapses consecutive duplicates)
# 3) App-split into train/test
# 4) Trains unigram + bigram + trigram (with smoothing)
# 5) Scores ALL REAL test transitions (no synthetic corruption)
# 6) Sorts by anomaly score, prints top-N, and exports CSV
#
# Optional (if you have screenshots): tries to resolve screenshot paths
# and prints them (you can open them manually).
#
# ============================================================

import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ----------------------------
# 0) CONFIG (EDIT THESE)
# ----------------------------
DATA_PATH = "processed_data/clusters/screen_clusters_k120.tsv"  # <-- your TSV
OUT_DIR = "anomaly_exports"
TOP_N_TO_EXPORT = 200
TOP_N_TO_PRINT = 30

# Model params (keep consistent with your report)
ALPHA_SMOOTH = 1.0
LAMBDA_BACKOFF = 0.2
EPS = 1e-12

# Trace preprocessing
COLLAPSE_CONSECUTIVE_DUPLICATES = True

# If you have a column that indicates ordering inside trace, set it here.
# If None, we assume TSV rows are already in order for each (app_id, trace_id).
ORDER_COL = None  # e.g., "screen_key" or "step_idx"

# Optional screenshot support:
# If you have a folder with screenshots per app/trace, set root here.
# Example expected layout (adjust as needed):
#   SCREENSHOT_ROOT/app_id/trace_id/screenshots/<screen_id>.png
SCREENSHOT_ROOT = None  # e.g., "/path/to/rico-like-dataset-root"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1) Load TSV
# ============================================================

def load_cluster_tsv(path: str) -> pd.DataFrame:
    print(f"[Load] Reading TSV: {path}")
    df = pd.read_csv(path, sep="\t")
    print(f"[Load] Rows: {len(df):,} | Cols: {len(df.columns)}")
    print(f"[Load] Columns: {list(df.columns)}")

    needed = {"app_id", "trace_id", "cluster_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if ORDER_COL is not None and ORDER_COL not in df.columns:
        raise ValueError(f"ORDER_COL='{ORDER_COL}' not found in TSV columns.")

    return df

df = load_cluster_tsv(DATA_PATH)

# ============================================================
# 2) Build traces (app_id, trace_id -> sequence of clusters)
# ============================================================

def build_traces(df: pd.DataFrame,
                 collapse_consecutive_duplicates: bool = True,
                 order_col: Optional[str] = None):
    traces = []
    grouped = df.groupby(["app_id", "trace_id"], sort=False)

    print("[Traces] Building sequences per (app_id, trace_id)...")
    n_total = 0
    n_kept = 0

    for (app_id, trace_id), g in grouped:
        if order_col is not None:
            g = g.sort_values(order_col, kind="mergesort")  # stable

        seq = g["cluster_id"].tolist()

        # Optional: keep screen_id list too for qualitative inspection
        screen_ids = g["screen_id"].tolist() if "screen_id" in g.columns else [None] * len(seq)

        if collapse_consecutive_duplicates and len(seq) > 1:
            collapsed_seq = [seq[0]]
            collapsed_sids = [screen_ids[0]]
            for x, sid in zip(seq[1:], screen_ids[1:]):
                if x != collapsed_seq[-1]:
                    collapsed_seq.append(x)
                    collapsed_sids.append(sid)
            seq = collapsed_seq
            screen_ids = collapsed_sids

        n_total += 1
        if len(seq) >= 2:
            traces.append((app_id, trace_id, seq, screen_ids))
            n_kept += 1

    print(f"[Traces] Total trace groups: {n_total:,}")
    print(f"[Traces] Kept traces (len>=2): {n_kept:,}")
    return traces

traces_raw = build_traces(df,
                          collapse_consecutive_duplicates=COLLAPSE_CONSECUTIVE_DUPLICATES,
                          order_col=ORDER_COL)

# ============================================================
# 3) Remap cluster IDs to 0..K-1 contiguous
# ============================================================

def remap_clusters(traces):
    all_ids = []
    for _, _, seq, _ in traces:
        all_ids.extend(seq)
    uniq = sorted(set(all_ids))
    id2new = {old:i for i, old in enumerate(uniq)}
    new2old = {i:old for old, i in id2new.items()}

    remapped = []
    for app_id, trace_id, seq, screen_ids in traces:
        remapped.append((app_id, trace_id, [id2new[x] for x in seq], screen_ids))

    K = len(uniq)
    print(f"[Clusters] Remapped cluster IDs to contiguous range 0..{K-1}")
    return remapped, id2new, new2old, K

traces, id2new, new2old, K = remap_clusters(traces_raw)

# ============================================================
# 4) App-split train/test
# ============================================================

def app_split(traces, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    apps = sorted(set(a for a, _, _, _ in traces))
    rng.shuffle(apps)
    n_test = int(round(len(apps) * test_frac))
    test_apps = set(apps[:n_test])

    train = [t for t in traces if t[0] not in test_apps]
    test  = [t for t in traces if t[0] in test_apps]

    print(f"[Split] Apps total: {len(apps):,}")
    print(f"[Split] Test apps:  {len(test_apps):,} ({test_frac*100:.1f}%)")
    print(f"[Split] Train traces: {len(train):,}")
    print(f"[Split] Test traces : {len(test):,}")
    return train, test, test_apps

train_traces, test_traces, test_apps = app_split(traces, test_frac=0.2, seed=42)

# ============================================================
# 5) Train N-gram models (Unigram/Bigram/Trigram)
# ============================================================

def train_unigram_next(train_traces, K: int, alpha: float = 1.0):
    counts = np.zeros(K, dtype=np.float64)
    for _, _, seq, _ in train_traces:
        for b in seq[1:]:
            counts[b] += 1.0
    counts += alpha
    probs = counts / counts.sum()
    return probs

def train_bigram_probs(train_traces, K: int, alpha: float = 1.0):
    counts = np.zeros((K, K), dtype=np.float64)
    for _, _, seq, _ in train_traces:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1.0
    counts += alpha
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs

def train_trigram_probs(train_traces, K: int, alpha: float = 1.0):
    counts = defaultdict(lambda: np.zeros(K, dtype=np.float64))
    for _, _, seq, _ in train_traces:
        if len(seq) < 3:
            continue
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            counts[(a, b)][c] += 1.0

    trigram_probs = {}
    for ctx, vec in counts.items():
        vec = vec + alpha
        trigram_probs[ctx] = vec / vec.sum()
    return trigram_probs

print("[Train] Training unigram/bigram/trigram ...")
unigram_probs = train_unigram_next(train_traces, K, alpha=ALPHA_SMOOTH)
bigram_probs  = train_bigram_probs(train_traces, K, alpha=ALPHA_SMOOTH)
trigram_probs = train_trigram_probs(train_traces, K, alpha=ALPHA_SMOOTH)

print(f"[Train] K={K} | trigram contexts={len(trigram_probs):,}")
print(f"[Train] unigram_probs shape={unigram_probs.shape} | bigram_probs shape={bigram_probs.shape}")

# ============================================================
# 6) Trigram backoff mixture + anomaly scoring
# ============================================================

def pmix_next(a: int, b: int,
              trigram_probs: Dict[Tuple[int,int], np.ndarray],
              bigram_probs: np.ndarray,
              unigram_probs: np.ndarray,
              lambda_backoff: float = 0.2) -> np.ndarray:
    """
    Returns probability vector over next cluster (size K).
    If trigram context exists: (1-l)*Ptri + l*Pbi
    Else: (1-l)*Pbi + l*Punigram
    """
    if (a, b) in trigram_probs:
        p_tri = trigram_probs[(a, b)]
        p_bi  = bigram_probs[b]
        return (1 - lambda_backoff) * p_tri + lambda_backoff * p_bi
    else:
        p_bi = bigram_probs[b]
        return (1 - lambda_backoff) * p_bi + lambda_backoff * unigram_probs

def anomaly_score(a: int, b: int, c: int, **kwargs) -> float:
    p = pmix_next(a, b, **kwargs)[c]
    return float(-math.log(p + EPS))

# ============================================================
# 7) Score ALL REAL test transitions and sort by anomaly score
# ============================================================

def guess_screenshot_path(app_id, trace_id, screen_id):
    if SCREENSHOT_ROOT is None or screen_id is None:
        return None
    # Try common patterns; you can edit this to match your dataset
    cand = os.path.join(SCREENSHOT_ROOT, str(app_id), str(trace_id), "screenshots", f"{screen_id}.png")
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(SCREENSHOT_ROOT, str(app_id), str(trace_id), "screenshots", f"{screen_id}.jpg")
    if os.path.exists(cand2):
        return cand2
    return None

print("[Score] Scoring real test transitions with trigram mixture ...")

rows = []
n_scored = 0
n_skipped_short = 0

for app_id, trace_id, seq, screen_ids in test_traces:
    if len(seq) < 3:
        n_skipped_short += 1
        continue

    for t in range(2, len(seq)):
        a, b, c = seq[t-2], seq[t-1], seq[t]

        score = anomaly_score(
            a, b, c,
            trigram_probs=trigram_probs,
            bigram_probs=bigram_probs,
            unigram_probs=unigram_probs,
            lambda_backoff=LAMBDA_BACKOFF
        )

        sid_prev2 = screen_ids[t-2] if t-2 < len(screen_ids) else None
        sid_prev1 = screen_ids[t-1] if t-1 < len(screen_ids) else None
        sid_next  = screen_ids[t]   if t   < len(screen_ids) else None

        rows.append({
            "app_id": app_id,
            "trace_id": trace_id,
            "t": t,
            "prev2_cluster": int(a),
            "prev1_cluster": int(b),
            "next_cluster": int(c),
            "prev2_cluster_orig": int(new2old[a]),
            "prev1_cluster_orig": int(new2old[b]),
            "next_cluster_orig": int(new2old[c]),
            "score_neglogp": float(score),
            "prev2_screen_id": sid_prev2,
            "prev1_screen_id": sid_prev1,
            "next_screen_id":  sid_next,
            "prev2_screenshot": guess_screenshot_path(app_id, trace_id, sid_prev2),
            "prev1_screenshot": guess_screenshot_path(app_id, trace_id, sid_prev1),
            "next_screenshot":  guess_screenshot_path(app_id, trace_id, sid_next),
        })
        n_scored += 1

print(f"[Score] Scored transitions: {n_scored:,}")
print(f"[Score] Skipped traces (len<3): {n_skipped_short:,}")

df_scores = pd.DataFrame(rows)
df_scores = df_scores.sort_values("score_neglogp", ascending=False).reset_index(drop=True)

print(f"[Score] Top score: {df_scores.loc[0, 'score_neglogp']:.4f}")
print(f"[Score] Median score: {df_scores['score_neglogp'].median():.4f}")
print(f"[Score] 95th pct score: {df_scores['score_neglogp'].quantile(0.95):.4f}")



# Keep only 1 example per (app_id) for diversity
df_unique_app = df_scores.drop_duplicates(subset=["app_id"], keep="first").reset_index(drop=True)

# Or stricter: 1 example per (app_id, trace_id)
df_unique_trace = df_scores.drop_duplicates(subset=["app_id", "trace_id"], keep="first").reset_index(drop=True)

print(f"[Dedupe] Unique by app_id:      {len(df_unique_app):,} examples")
print(f"[Dedupe] Unique by app+trace:  {len(df_unique_trace):,} examples")


# ============================================================
# 8) Print Top-N to inspect manually
# ============================================================

def print_top(df_scores: pd.DataFrame, n: int = 30):
    n = min(n, len(df_scores))
    print("\n" + "="*80)
    print(f"[Top {n}] Highest anomaly-score REAL transitions (no corruption)")
    print("="*80)

    for i in range(n):
        r = df_scores.iloc[i]
        print(f"\n#{i+1:02d} | score={r['score_neglogp']:.4f} | app={r['app_id']} | trace={r['trace_id']} | t={r['t']}")
        print(f"   clusters (remapped): ({r['prev2_cluster']}, {r['prev1_cluster']}) -> {r['next_cluster']}")
        print(f"   clusters (orig ids): ({r['prev2_cluster_orig']}, {r['prev1_cluster_orig']}) -> {r['next_cluster_orig']}")
        if r["prev2_screen_id"] is not None:
            print(f"   screen_ids: prev2={r['prev2_screen_id']} | prev1={r['prev1_screen_id']} | next={r['next_screen_id']}")
        if SCREENSHOT_ROOT is not None:
            print(f"   screenshots:")
            print(f"     prev2: {r['prev2_screenshot']}")
            print(f"     prev1: {r['prev1_screenshot']}")
            print(f"     next : {r['next_screenshot']}")

print_top(df_scores, TOP_N_TO_PRINT)

# ============================================================
# 9) Export CSV for curation (top-N)
# ============================================================

export_path = os.path.join(OUT_DIR, f"top_anomalies_trigram_lambda{LAMBDA_BACKOFF}_top{TOP_N_TO_EXPORT}.csv")
df_scores.head(TOP_N_TO_EXPORT).to_csv(export_path, index=False)
print("\n" + "="*80)
print(f"[Export] Saved top-{TOP_N_TO_EXPORT} anomalies to:")
print(f"         {export_path}")
print("="*80)

# ============================================================
# 10) (Optional) Export also "Top-5 predicted next roles" per example
#     This helps you fill the figure template.
# ============================================================

def topk_predictions(a, b, k=5):
    probs = pmix_next(
        a, b,
        trigram_probs=trigram_probs,
        bigram_probs=bigram_probs,
        unigram_probs=unigram_probs,
        lambda_backoff=LAMBDA_BACKOFF
    )
    idx = np.argsort(-probs)[:k]
    return [(int(i), float(probs[i]), int(new2old[i])) for i in idx]

print("\n[Pred] Adding Top-5 predicted next clusters for the TOP printed examples...")
pred_rows = []
for i in range(min(TOP_N_TO_PRINT, len(df_scores))):
    r = df_scores.iloc[i]
    a = int(r["prev2_cluster"])
    b = int(r["prev1_cluster"])
    preds = topk_predictions(a, b, k=5)

    pred_rows.append({
        "rank": i+1,
        "app_id": r["app_id"],
        "trace_id": r["trace_id"],
        "t": r["t"],
        "score_neglogp": r["score_neglogp"],
        "context_prev2_cluster": a,
        "context_prev1_cluster": b,
        "true_next_cluster": int(r["next_cluster"]),
        "pred1_cluster": preds[0][0], "pred1_prob": preds[0][1], "pred1_cluster_orig": preds[0][2],
        "pred2_cluster": preds[1][0], "pred2_prob": preds[1][1], "pred2_cluster_orig": preds[1][2],
        "pred3_cluster": preds[2][0], "pred3_prob": preds[2][1], "pred3_cluster_orig": preds[2][2],
        "pred4_cluster": preds[3][0], "pred4_prob": preds[3][1], "pred4_cluster_orig": preds[3][2],
        "pred5_cluster": preds[4][0], "pred5_prob": preds[4][1], "pred5_cluster_orig": preds[4][2],
    })

pred_df = pd.DataFrame(pred_rows)
pred_export_path = os.path.join(OUT_DIR, f"top{TOP_N_TO_PRINT}_predictions_trigram_lambda{LAMBDA_BACKOFF}.csv")
pred_df.to_csv(pred_export_path, index=False)

print(f"[Pred] Saved Top-{TOP_N_TO_PRINT} predictions table to:")
print(f"       {pred_export_path}")

print("\nDone. Next: open the exported CSV(s), inspect Top-30, pick 5–8 diverse examples for figures.")
