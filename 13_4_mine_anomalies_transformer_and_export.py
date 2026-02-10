import os
import math
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import torch
import torch.nn as nn


# -----------------------------
# Utilities: sequence building
# -----------------------------
def parse_screen_id_for_sort(s):
    try:
        return int(str(s).split("_")[0])
    except Exception:
        return str(s)

def build_sequences_with_meta(clusters_tsv: str, collapse_consecutive=True):
    """
    Returns sequences per (app_id, trace_id) with aligned metadata:
      - clusters: [c0, c1, ...]
      - screen_ids: [sid0, sid1, ...]
      - screen_keys: [sk0, sk1, ...]
    """
    df = pd.read_csv(clusters_tsv, sep="\t")
    required = {"app_id", "trace_id", "screen_id", "cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"clusters_tsv missing columns: {missing}")

    # screen_key optional but recommended
    has_screen_key = "screen_key" in df.columns

    seqs = []
    for (app, trace), g in df.groupby(["app_id", "trace_id"]):
        g2 = g.copy()
        g2["sort_key"] = g2["screen_id"].apply(parse_screen_id_for_sort)
        g2 = g2.sort_values("sort_key")

        clusters = g2["cluster_id"].astype(int).tolist()
        screen_ids = g2["screen_id"].astype(str).tolist()
        screen_keys = g2["screen_key"].astype(str).tolist() if has_screen_key else [""] * len(clusters)

        if collapse_consecutive and len(clusters) >= 2:
            c2, sid2, sk2 = [clusters[0]], [screen_ids[0]], [screen_keys[0]]
            for c, sid, sk in zip(clusters[1:], screen_ids[1:], screen_keys[1:]):
                if c != c2[-1]:
                    c2.append(c); sid2.append(sid); sk2.append(sk)
            clusters, screen_ids, screen_keys = c2, sid2, sk2

        if len(clusters) >= 2:
            seqs.append((app, trace, clusters, screen_ids, screen_keys))

    return seqs, df


def get_num_clusters_from_df(df: pd.DataFrame) -> int:
    mx = int(df["cluster_id"].max())
    return mx + 1


def build_cluster_representatives(df: pd.DataFrame):
    """
    Choose a representative screen per cluster_id and KEEP its app/trace/screen_id,
    so exp_img can point to the correct screenshot path.
    """
    reps = {}
    for cid, g in df.groupby("cluster_id"):
        # pick most frequent screen_key if available, else most frequent (app,trace,screen_id) tuple
        if "screen_key" in g.columns:
            sk = g["screen_key"].astype(str).value_counts().idxmax()
            gk = g[g["screen_key"].astype(str) == sk]
            row = gk.iloc[0]
        else:
            # fallback: pick the most frequent screen_id row
            sid = g["screen_id"].astype(str).value_counts().idxmax()
            gk = g[g["screen_id"].astype(str) == sid]
            row = gk.iloc[0]
            sk = ""

        reps[int(cid)] = {
            "app_id": str(row["app_id"]),
            "trace_id": str(row["trace_id"]),
            "screen_id": str(row["screen_id"]),
            "screen_key": str(sk),
        }
    return reps



def build_cluster_representatives_old(df: pd.DataFrame):
    """
    Choose a representative (screen_id, screen_key) per cluster_id.
    Uses the most frequent screen_key if available, else most frequent screen_id.
    """
    reps = {}
    if "screen_key" in df.columns:
        for cid, g in df.groupby("cluster_id"):
            # most frequent screen_key
            sk = g["screen_key"].astype(str).value_counts().idxmax()
            gk = g[g["screen_key"].astype(str) == sk]
            # choose a screen_id among those (most frequent)
            sid = gk["screen_id"].astype(str).value_counts().idxmax()
            reps[int(cid)] = {"screen_id": str(sid), "screen_key": str(sk)}
    else:
        for cid, g in df.groupby("cluster_id"):
            sid = g["screen_id"].astype(str).value_counts().idxmax()
            reps[int(cid)] = {"screen_id": str(sid), "screen_key": ""}
    return reps


# -----------------------------
# Model: must match your transformer class
# -----------------------------
class CausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_ctx: int, d_model=256, nhead=4, num_layers=4, dropout=0.1, pad_id=0):
        super().__init__()
        self.max_ctx = max_ctx
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_ctx, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, attn_mask):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        h = self.tok(x) + self.pos(pos_ids)
        h = h.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        key_pad = ~attn_mask
        h = self.enc(h, mask=causal, src_key_padding_mask=key_pad)
        h = h.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        lengths = attn_mask.long().sum(dim=1)
        last_idx = lengths - 1
        last_h = h[torch.arange(B, device=x.device), last_idx]
        return self.head(last_h)


def load_transformer_from_ckpt(ckpt_path: str, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    K = int(ckpt["K"])
    pad_id = int(ckpt["pad_id"])
    bos_id = int(ckpt["bos_id"])
    cfg = ckpt.get("cfg", {})

    max_ctx = int(cfg.get("max_ctx", 16))
    d_model = int(cfg.get("d_model", 256))
    nhead = int(cfg.get("nhead", 4))
    num_layers = int(cfg.get("num_layers", 4))
    dropout = float(cfg.get("dropout", 0.1))

    vocab = K + 2

    model = CausalTransformer(
        vocab_size=vocab,
        max_ctx=max_ctx,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        pad_id=pad_id,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    return model, K, pad_id, bos_id, max_ctx


# -----------------------------
# Image path helper
# -----------------------------
def format_img_path(pattern: str, app_id: str, trace_id: str, screen_id: str, screen_key: str):
    """
    pattern examples:
      "screens/{app_id}/{trace_id}/{screen_id}.png"
      "screens/{app_id}/{screen_id}.jpg"
      "screens/{screen_key}.png"
    """
    return pattern.format(app_id=app_id, trace_id=trace_id, screen_id=screen_id, screen_key=screen_key)


def build_cluster_app_index(df: pd.DataFrame):
    """
    index[(cluster_id, app_id)] -> list of (trace_id, screen_id, screen_key)
    """
    idx = defaultdict(list)
    has_sk = "screen_key" in df.columns
    for _, r in df.iterrows():
        cid = int(r["cluster_id"])
        app = str(r["app_id"])
        idx[(cid, app)].append((
            str(r["trace_id"]),
            str(r["screen_id"]),
            str(r["screen_key"]) if has_sk else ""
        ))
    return idx



# -----------------------------
# Main: mine anomalies + export
# -----------------------------
@torch.no_grad()
def mine_anomalies_transformer(
    clusters_tsv: str,
    ckpt_path: str,
    out_tsv: str,
    img_pattern: str,
    device="cuda",
    trigram_subset=True,
    top_n=200,
    topk_preds=5,
    eps=1e-12,
):
    """
    Writes a TSV containing anomaly cases with:
      prev screen, observed next, expected next (rep), scores, etc.

    trigram_subset=True:
      require two REAL previous clusters before evaluating an edge.
      With BOS prepended, that means start at t=3 for seq_b.
    """
    seqs, df = build_sequences_with_meta(clusters_tsv, collapse_consecutive=True)
    K = get_num_clusters_from_df(df)
    reps = build_cluster_representatives(df)

    
    # Expected screen = a representative from the same app (or at least same trace) if available
    cluster_app_index = build_cluster_app_index(df)

    model, K_ckpt, pad_id, bos_id, max_ctx = load_transformer_from_ckpt(ckpt_path, device=device)
    if K_ckpt != K:
        print(f"[Warn] K in data ({K}) != K in checkpoint ({K_ckpt}). Using checkpoint K for logits slice.")
        K = K_ckpt

    rows = []

    # Start position definition
    # seq_b = [BOS] + [c0,c1,c2,...]
    # trigram_subset requires two REAL prev clusters => first predicted real token is c2? No.
    # To predict c2 you only have one real prev (c0,c1?) -> Actually c2 needs c0,c1. That corresponds to seq_b index t=3 predicting c2.
    start_t = 3 if trigram_subset else 1

    for app_id, trace_id, clusters, screen_ids, screen_keys in seqs:
        # prepend BOS to cluster sequence; keep screen meta aligned to real tokens only
        seq_b = [bos_id] + list(clusters)

        # if trigram_subset, require original len(clusters) >= 3
        if trigram_subset and len(clusters) < 3:
            continue

        for t in range(start_t, len(seq_b)):
            # t indexes into seq_b, and corresponds to real token index (t-1) in clusters/screen_ids
            true_next = seq_b[t]

            # prefix includes BOS + previous reals
            prefix = seq_b[:t]

            T = min(max_ctx, len(prefix))
            x = torch.full((1, T), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((1, T), dtype=torch.bool, device=device)

            p = prefix[-T:]
            L = len(p)
            x[0, :L] = torch.tensor(p, dtype=torch.long, device=device)
            attn[0, :L] = True

            logits = model(x, attn)[0]              # (V,)
            probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
            probs_clusters = probs[:K]              # only real clusters

            p_obs = float(probs_clusters[true_next])
            score = -math.log(max(p_obs, eps))

            expected = int(np.argmax(probs_clusters))
            p_exp = float(probs_clusters[expected])
            margin = math.log(max(p_exp, eps)) - math.log(max(p_obs, eps))

            # Map indices to screens:
            # current step in real seq is (t-1)
            idx_next = t - 1
            idx_prev = idx_next - 1

            prev_sid = screen_ids[idx_prev]
            prev_sk = screen_keys[idx_prev]
            obs_sid = screen_ids[idx_next]
            obs_sk = screen_keys[idx_next]


            ## before fix 1
            #rep = reps.get(expected, {"screen_id": "", "screen_key": ""})
            #exp_sid = rep["screen_id"]
            #exp_sk = rep["screen_key"]


            ## before fix 2
            """
            rep = reps.get(expected, {"app_id": "", "trace_id": "", "screen_id": "", "screen_key": ""})

            exp_app = rep["app_id"]
            exp_trace = rep["trace_id"]
            exp_sid = rep["screen_id"]
            exp_sk = rep["screen_key"]
            """


            # --- expected representative: prefer same-app, else fallback to global ---
            same_app_list = cluster_app_index.get((expected, app_id), [])

            if len(same_app_list) > 0:
                # pick the most frequent within same-app (simple pick first; can improve to most common)
                exp_trace, exp_sid, exp_sk = same_app_list[0]
                exp_app = app_id
                exp_choice = "same_app"
            else:
                rep = reps.get(expected, {"app_id": "", "trace_id": "", "screen_id": "", "screen_key": ""})
                exp_app = rep["app_id"]
                exp_trace = rep["trace_id"]
                exp_sid = rep["screen_id"]
                exp_sk = rep["screen_key"]
                exp_choice = "global"

            row = {
                "app_id": app_id,
                "trace_id": trace_id,
                "t_real": idx_next,  # index in real cluster seq
                "prev_cluster": int(clusters[idx_prev]),
                "obs_cluster": int(clusters[idx_next]),
                "exp_cluster": int(expected),

                "p_obs": p_obs,
                "p_exp": p_exp,
                "anomaly_score": score,
                "margin_log": margin,

                "prev_screen_id": prev_sid,
                "obs_screen_id": obs_sid,
                "exp_rep_screen_id": exp_sid,

                "prev_screen_key": prev_sk,
                "obs_screen_key": obs_sk,
                "exp_rep_screen_key": exp_sk,

                "prev_img": format_img_path(img_pattern, app_id, trace_id, prev_sid, prev_sk),
                "obs_img": format_img_path(img_pattern, app_id, trace_id, obs_sid, obs_sk),
                "exp_img": format_img_path(img_pattern, app_id, trace_id, exp_sid, exp_sk),
            }
            
            ## before fix 2
            ##row["exp_rep_app_id"] = exp_app
            ##row["exp_rep_trace_id"] = exp_trace
            #row["exp_img"] = format_img_path(img_pattern, exp_app, exp_trace, exp_sid, exp_sk)

            row["exp_choice"] = exp_choice
            row["exp_rep_app_id"] = exp_app
            row["exp_rep_trace_id"] = exp_trace
            row["exp_rep_screen_id"] = exp_sid
            row["exp_rep_screen_key"] = exp_sk

            row["exp_img"] = format_img_path(img_pattern, exp_app, exp_trace, exp_sid, exp_sk)


            if topk_preds and topk_preds > 0:
                topk = np.argsort(-probs_clusters)[:topk_preds]
                row["topk_pred_clusters"] = json.dumps([int(i) for i in topk])
                row["topk_pred_probs"] = json.dumps([float(probs_clusters[i]) for i in topk])

            rows.append(row)

    out_df = pd.DataFrame(rows)

    # Select top anomalies: by anomaly_score (largest) then margin
    out_df = out_df.sort_values(["anomaly_score", "margin_log"], ascending=[False, False])

    if top_n is not None and top_n > 0:
        out_df = out_df.head(top_n).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[Saved] {out_tsv}  rows={len(out_df)}")
    return out_df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("clusters_tsv", help="processed_data/clusters/screen_clusters_kXX.tsv")
    ap.add_argument("--ckpt", required=True, help="transformer checkpoint .pt")
    ap.add_argument("--out_tsv", default="anomalies_transformer_top.tsv")
    ap.add_argument("--img_pattern", required=True,
                    help='e.g. "screens/{app_id}/{trace_id}/{screen_id}.png" or "screens/{screen_key}.png"')
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top_n", type=int, default=200)
    ap.add_argument("--topk_preds", type=int, default=5)
    ap.add_argument("--subset", action="store_true", help="use trigram_subset definition (two real prev)")
    args = ap.parse_args()

    mine_anomalies_transformer(
        clusters_tsv=args.clusters_tsv,
        ckpt_path=args.ckpt,
        out_tsv=args.out_tsv,
        img_pattern=args.img_pattern,
        device=args.device,
        trigram_subset=args.subset,
        top_n=args.top_n,
        topk_preds=args.topk_preds,
    )


    """


    Case 1: screenshots stored by screen_id inside app/trace folders


python 13_4_mine_anomalies_transformer_and_export.py \
processed_data/clusters/screen_clusters_k40.tsv \
--ckpt experiments_weights/best_val_0.7187_transformer_next_cluster_16_200_0.2_8_0.3_256_3e-05_0.1_0.0.pt \
--out_tsv outputs/anomalies_tf_k40_top200.tsv \
--img_pattern "../../dataset/traces/filtered_traces/{app_id}/{trace_id}/screenshots/{screen_id}.jpg" \
--subset \
--top_n 200 \
--topk_preds 5

Case 2: screenshots named by screen_key
Better one for me

python 13_4_mine_anomalies_transformer_and_export.py \
processed_data/clusters/screen_clusters_k40.tsv \
--ckpt experiments_weights/best_val_0.7187_transformer_next_cluster_16_200_0.2_8_0.3_256_3e-05_0.1_0.0.pt \
--out_tsv outputs/anomalies_tf_k40_top200.tsv \
--img_pattern "screens/{screen_key}.png" \
--subset
   
    
    """



## and to make sure from the results, we plot it as follows in visualize_with_transformers.ipynb:
 ### in  aseparate file visualize_with_trasnformers.ipynb 
"""
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import re

def show_anomaly_row(row):
    imgs = [row["prev_img"], row["obs_img"], row["exp_img"]]
    titles = [
        f"Source\nprev_cluster={row['prev_cluster']}",
        f"Observed (Anomaly)\nobs_cluster={row['obs_cluster']}\np_obs={row['p_obs']:.3g}\nscore={row['anomaly_score']:.2f}",
        f"Expected (Model)\nexp_cluster={row['exp_cluster']}\np_exp={row['p_exp']:.3g}\nmargin={row['margin_log']:.2f}",
    ]

    plt.figure(figsize=(12,4))
    for i, (path, title) in enumerate(zip(imgs, titles), 1):
        ax = plt.subplot(1,3,i)
        ax.set_title(title)
        ax.axis("off")
        try:
            #path = re.sub(r'/(trace_\d+)/(\d+)\.png', r'/\1/screenshots/\2.jpg', path)
            #path = path.replace("screens/","../../dataset/traces/filtered_traces/")
           
            print("after changed",path)
            ax.imshow(Image.open(path).convert("RGB"))
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load:\n{path}\n\n{e}", ha="center", va="center")
    plt.tight_layout()
    plt.show()

df = pd.read_csv("outputs/anomalies_tf_k40_top200.tsv", sep="\t")


print("all exp choices:",df["exp_choice"].value_counts())

# --- Paper-figure curation filters (apply here) ---
df = df[
    (df["p_obs"] < 0.01) &
    (df["p_exp"] > 0.3) &
    (df["margin_log"] > 4.0)
].copy()

# Optional: avoid cross-app "expected" screens for figure clarity
if "exp_rep_app_id" in df.columns:
    df = df[df["app_id"] == df["exp_rep_app_id"]].copy()

df = df.sort_values(["anomaly_score", "margin_log"], ascending=[False, False]).reset_index(drop=True)

print("After filters:", len(df))

# Render top N after filtering
for i in range(min(10, len(df))):
    show_anomaly_row(df.iloc[i])

    


"""