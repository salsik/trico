
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Shared preprocessing
# -----------------------------
def parse_screen_id_for_sort(s):
    try:
        return int(str(s).split("_")[0])
    except Exception:
        return str(s)


def build_sequences(clusters_tsv: str) -> List[Tuple[str, str, List[int]]]:
    df = pd.read_csv(clusters_tsv, sep="\t")
    seqs = []
    for (app, trace), g in df.groupby(["app_id", "trace_id"]):
        g2 = g.copy()
        g2["sort_key"] = g2["screen_id"].apply(parse_screen_id_for_sort)
        g2 = g2.sort_values("sort_key")
        seq = g2["cluster_id"].astype(int).tolist()

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


def sanity_check_clusters(seqs, K: int):
    all_ids = []
    for _, _, seq in seqs:
        all_ids.extend(seq)
    mn, mx = min(all_ids), max(all_ids)
    print(f"[Sanity] cluster_id min={mn} max={mx} K={K}")
    assert mn >= 0, "Found negative cluster_id (e.g., -1)."
    assert mx < K, "Found cluster_id >= K."


# -----------------------------
# Baselines from original script
# -----------------------------
def recall_from_ranked(true_next, ranked, k):
    return 1 if true_next in ranked[:k] else 0


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
    tri_counts = defaultdict(Counter)
    bi_counts = np.zeros((K, K), dtype=np.float64)
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

    tri_top = [x for x, _ in counter.most_common(max(k * 5, 50))]
    bi_top = np.argsort(-bi_probs[c])[:max(k * 5, 50)].tolist()
    cand = list(dict.fromkeys(tri_top + bi_top))

    scores = []
    for n in cand:
        p_tri = counter[n] / total
        p_bi = bi_probs[c, n]
        scores.append((1.0 - lambda_backoff) * p_tri + lambda_backoff * p_bi)

    idx = np.argsort(-np.array(scores))[:k]
    return np.array([cand[i] for i in idx], dtype=np.int64)


def eval_unigram_trigram_subset(test_seqs, probs, k_list=(1, 3, 5, 10)):
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
    return {k: hits[k] / max(1, total) for k in k_list}, total


def eval_1gram_trigram_subset(test_seqs, probs, k_list=(1, 3, 5, 10)):
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
    return {k: hits[k] / max(1, total) for k in k_list}, total


def eval_2gram_trigram_subset(test_seqs, tri_counts, bi_probs, k_list=(1, 3, 5, 10), lambda_backoff=0.2):
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


# -----------------------------
# Transformer components
# -----------------------------
class PrefixNextDataset(Dataset):
    def __init__(self, seqs: List[Tuple[str, str, List[int]]], max_ctx: int, pad_id: int, bos_id: int):
        self.max_ctx = max_ctx
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.samples = []

        for _, _, seq in seqs:
            seq = [bos_id] + list(seq)
            for t in range(1, len(seq)):
                prefix = seq[:t]
                nxt = seq[t]
                self.samples.append((prefix, nxt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate(self, batch):
        prefixes, targets = zip(*batch)
        B = len(prefixes)
        T = min(self.max_ctx, max(len(p) for p in prefixes))

        x = torch.full((B, T), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.bool)

        for i, p in enumerate(prefixes):
            p = p[-T:]
            L = len(p)
            x[i, :L] = torch.tensor(p, dtype=torch.long)
            attn[i, :L] = True

        y = torch.tensor(targets, dtype=torch.long)
        return x, attn, y


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_ctx: int, d_model=256, nhead=4, num_layers=4, dropout=0.1, pad_id=0):
        super().__init__()
        self.max_ctx = max_ctx
        self.pad_id = pad_id

        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_ctx, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
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
        if (lengths <= 0).any():
            raise RuntimeError("Empty (all-pad) sequence encountered.")
        last_idx = lengths - 1

        last_h = h[torch.arange(B, device=x.device), last_idx]
        logits = self.head(last_h)
        return logits


@torch.no_grad()
def eval_model_on_edges(model, test_seqs, device, max_ctx, pad_id, bos_id, ks=(1, 3, 5, 10), trigram_subset=False):
    model.eval()
    hits = {k: 0 for k in ks}
    total = 0

    for _, _, seq in test_seqs:
        seq = [bos_id] + list(seq)
        start_t = 2 if trigram_subset else 1

        for t in range(start_t, len(seq)):
            prefix = seq[:t]
            target = seq[t]

            T = min(max_ctx, len(prefix))
            x = torch.full((1, T), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((1, T), dtype=torch.bool, device=device)

            p = prefix[-T:]
            L = len(p)
            x[0, :L] = torch.tensor(p, dtype=torch.long, device=device)
            attn[0, :L] = True

            logits = model(x, attn)
            ranked = torch.argsort(logits[0], descending=True).detach().cpu().numpy()

            total += 1
            for k in ks:
                if target in ranked[:k]:
                    hits[k] += 1

    scores = {k: hits[k] / max(1, total) for k in ks}
    return scores, total


@dataclass
class TrainCfg:
    max_ctx: int = 16
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    epochs: int = 50
    seed: int = 42
    val_ratio: float = 0.1
    patience: int = 8
    min_delta: float = 1e-4


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val_by_app(train_seqs, val_ratio=0.1, seed=42):
    apps = sorted(list({a for a, _, _ in train_seqs}))
    rng = np.random.default_rng(seed)
    rng.shuffle(apps)
    n_val = max(1, int(len(apps) * val_ratio))

    if len(apps) <= 1:
        return train_seqs, [], set()
    if n_val >= len(apps):
        n_val = len(apps) - 1

    val_apps = set(apps[:n_val])
    tr = [s for s in train_seqs if s[0] not in val_apps]
    va = [s for s in train_seqs if s[0] in val_apps]
    return tr, va, val_apps


def train_transformer(train_seqs, val_seqs, K: int, cfg: TrainCfg, device: str):
    pad_id = K
    bos_id = K + 1
    vocab = K + 2

    ds = PrefixNextDataset(train_seqs, max_ctx=cfg.max_ctx, pad_id=pad_id, bos_id=bos_id)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=ds.collate)

    model = CausalTransformer(
        vocab_size=vocab,
        max_ctx=cfg.max_ctx,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_id=pad_id,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best_val = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    bad_epochs = 0
    ks = (1, 3, 5, 10)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for x, attn, y in dl:
            x, attn, y = x.to(device), attn.to(device), y.to(device)
            logits = model(x, attn)
            loss = loss_fn(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during training")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total_loss / max(1, n)

        if len(val_seqs) > 0:
            val_scores, val_edges = eval_model_on_edges(
                model, val_seqs, device, cfg.max_ctx, pad_id, bos_id, ks=ks, trigram_subset=True
            )
            val_r10 = val_scores[10]
            print(
                f"[Epoch {ep}/{cfg.epochs}] train_loss={train_loss:.4f} | "
                f"val_edges={val_edges} | val R@10(subset)={val_r10:.4f}"
            )

            if val_r10 > best_val + cfg.min_delta:
                best_val = val_r10
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
                print(f"  [Best] best_val R@10(subset)={best_val:.4f}")
            else:
                bad_epochs += 1
                print(f"  [No improve] bad_epochs={bad_epochs}/{cfg.patience}")
                if bad_epochs >= cfg.patience:
                    print("[EarlyStop] stopping.")
                    break
        else:
            print(f"[Epoch {ep}/{cfg.epochs}] train_loss={train_loss:.4f} | no validation split available")
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = float("nan")

    if best_state:
        model.load_state_dict(best_state)
        print(f"[Restore] Loaded best checkpoint with val R@10(subset)={best_val}")

    return model, pad_id, bos_id, best_val


def main(args):
    cfg = TrainCfg(
        max_ctx=args.ctx,
        epochs=args.epochs,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        patience=args.patience,
        min_delta=args.min_delta,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    k_list = (1, 3, 5, 10)

    seqs = build_sequences(args.clusters_tsv)
    K = get_num_clusters(seqs)
    sanity_check_clusters(seqs, K)

    with open(args.app_category_tsv, "r", encoding="utf-8", errors="replace") as f:
        cat_df = pd.read_csv(f, low_memory=False)


    #cat_df = pd.read_csv(args.app_category_tsv, encoding="utf-8", errors="replace", low_memory=False)
    #cat_df = pd.read_csv(args.app_category_tsv, encoding="utf-8", low_memory=False)
    cat_map = dict(zip(cat_df["App_Package_Name"].astype(str), cat_df["Category"].astype(str)))

    seqs2 = []
    for app, trace, seq in seqs:
        cat = cat_map.get(str(app), None)
        if cat is not None:
            seqs2.append((app, trace, seq, cat))

    apps_by_cat = defaultdict(set)
    for app, _, _, cat in seqs2:
        apps_by_cat[cat].add(app)

    cats = [c for c in sorted(apps_by_cat.keys()) if len(apps_by_cat[c]) >= args.min_apps_per_category]
    print(
        f"K={K} | total traces={len(seqs2)} | categories used={len(cats)} "
        f"(min_apps_per_category={args.min_apps_per_category})"
    )

    rows = []
    for idx, heldout in enumerate(cats, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(cats)}] Held-out category: {heldout}")

        train = [(a, t, s) for (a, t, s, c) in seqs2 if c != heldout]
        test = [(a, t, s) for (a, t, s, c) in seqs2 if c == heldout]
        train_tr, val, val_apps = split_train_val_by_app(train, val_ratio=cfg.val_ratio, seed=cfg.seed)

        print(
            f"[Split] heldout_apps={len(apps_by_cat[heldout])} | train_traces={len(train)} | "
            f"inner_train={len(train_tr)} | val={len(val)} | test={len(test)} | val_apps={len(val_apps)}"
        )

        uni = train_unigram_next(train, K, alpha=args.alpha_smooth)
        p1 = train_1gram(train, K, alpha=args.alpha_smooth)
        tri_counts, bi_counts = train_2gram(train, K)
        bi_probs = bi_counts + args.alpha_smooth
        bi_probs = bi_probs / bi_probs.sum(axis=1, keepdims=True)

        uni_sc, e_uni = eval_unigram_trigram_subset(test, uni, k_list=k_list)
        p1_sc, e_1 = eval_1gram_trigram_subset(test, p1, k_list=k_list)
        p2_sc, e_2 = eval_2gram_trigram_subset(
            test, tri_counts, bi_probs, k_list=k_list, lambda_backoff=args.lambda_backoff
        )

        tf_model, pad_id, bos_id, best_val = train_transformer(train_tr, val, K, cfg, device)
        tf_sc, e_tf = eval_model_on_edges(
            tf_model, test, device, cfg.max_ctx, pad_id, bos_id, ks=k_list, trigram_subset=True
        )

        row = {
            "heldout_category": heldout,
            "test_apps": len(apps_by_cat[heldout]),
            "test_traces": len(test),
            "edges_eval": e_2,
            "tf_edges_eval": e_tf,
            "tf_best_val_r10_subset": best_val,
            "uni_R1": uni_sc[1], "uni_R3": uni_sc[3], "uni_R5": uni_sc[5], "uni_R10": uni_sc[10],
            "1g_R1": p1_sc[1], "1g_R3": p1_sc[3], "1g_R5": p1_sc[5], "1g_R10": p1_sc[10],
            "2g_R1": p2_sc[1], "2g_R3": p2_sc[3], "2g_R5": p2_sc[5], "2g_R10": p2_sc[10],
            "tf_R1": tf_sc[1], "tf_R3": tf_sc[3], "tf_R5": tf_sc[5], "tf_R10": tf_sc[10],
        }
        rows.append(row)

        print(f"Held-out: {heldout} | test_apps={len(apps_by_cat[heldout])} | edges={e_2} | tf_edges={e_tf}")
        print(f"  Unigram     R@1={uni_sc[1]:.4f} R@3={uni_sc[3]:.4f} R@5={uni_sc[5]:.4f} R@10={uni_sc[10]:.4f}")
        print(f"  1-gram      R@1={p1_sc[1]:.4f} R@3={p1_sc[3]:.4f} R@5={p1_sc[5]:.4f} R@10={p1_sc[10]:.4f}")
        print(f"  2-gram      R@1={p2_sc[1]:.4f} R@3={p2_sc[3]:.4f} R@5={p2_sc[5]:.4f} R@10={p2_sc[10]:.4f}")
        print(f"  Transformer R@1={tf_sc[1]:.4f} R@3={tf_sc[3]:.4f} R@5={tf_sc[5]:.4f} R@10={tf_sc[10]:.4f}")

        if args.save_checkpoints:
            ckpt = {
                "state_dict": tf_model.state_dict(),
                "K": K,
                "pad_id": pad_id,
                "bos_id": bos_id,
                "cfg": cfg.__dict__,
                "heldout_category": heldout,
                "val_best_metric": "R@10(subset)",
                "val_best_value": best_val,
            }
            safe_cat = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(heldout))
            ckpt_path = f"{args.checkpoint_dir}/transformer_leaveout_{safe_cat}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  [Saved checkpoint] {ckpt_path}")

        del tf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pd.DataFrame(rows).to_csv(args.out_tsv, sep="\t", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\nSaved: {args.out_tsv}")

    if len(out) > 0:
        print("\n=== Macro avg over held-out categories ===")
        for prefix in ["uni", "1g", "2g", "tf"]:
            r1 = out[f"{prefix}_R1"].mean()
            r3 = out[f"{prefix}_R3"].mean()
            r5 = out[f"{prefix}_R5"].mean()
            r10 = out[f"{prefix}_R10"].mean()
            print(f"{prefix:>4s}  R@1={r1:.4f} R@3={r3:.4f} R@5={r5:.4f} R@10={r10:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters_tsv", type=str, default="processed_data/clusters/screen_clusters_k40.tsv")
    ap.add_argument("--app_category_tsv", type=str, default="../../dataset/6- app_details.csv")
    ap.add_argument("--out_tsv", type=str, default="category_leave_one_out_results_with_transformer_retrain.tsv")
    ap.add_argument("--min_apps_per_category", type=int, default=50)
    ap.add_argument("--lambda_backoff", type=float, default=0.5)
    ap.add_argument("--alpha_smooth", type=float, default=1.0)

    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-05)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--save_checkpoints", action="store_true")
    ap.add_argument("--checkpoint_dir", type=str, default="experiments_weights")

    args = ap.parse_args()
    main(args)









"""
run it as follow

python 11_2_category_leave_one_out_transformer_retrain.py \
  --clusters_tsv processed_data/clusters/screen_clusters_k40.tsv \
  --app_category_tsv "../../dataset/6- app_details.csv" \
  --ctx 16 \
  --epochs 200 \
  --patience 8 \
  --val_ratio 0.2 \
  --lr 3e-05 \
  --dropout 0.3 \
  --label_smoothing 0.0 \
  --weight_decay 0.1 \
  --batch_size 256



"""