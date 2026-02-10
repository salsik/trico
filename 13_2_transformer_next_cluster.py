# 13_transformer_next_cluster.py
import argparse
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Preprocessing (same style as your baselines)
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


def split_train_val_by_app(train_seqs, val_ratio=0.1, seed=42):
    apps = sorted(list({a for a, _, _ in train_seqs}))
    rng = np.random.default_rng(seed)
    rng.shuffle(apps)
    n_val = max(1, int(len(apps) * val_ratio))
    val_apps = set(apps[:n_val])

    tr = [s for s in train_seqs if s[0] not in val_apps]
    va = [s for s in train_seqs if s[0] in val_apps]
    return tr, va, val_apps


def get_num_clusters(seqs) -> int:
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
# Dataset: prefix -> next token (RIGHT padding)
# -----------------------------
class PrefixNextDataset(Dataset):
    def __init__(self, seqs: List[Tuple[str, str, List[int]]], max_ctx: int, pad_id: int, bos_id: int):
        self.max_ctx = max_ctx
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.samples = []  # (prefix_list, next_token)

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
            # RIGHT padding: tokens at beginning, pad at end
            x[i, :L] = torch.tensor(p, dtype=torch.long)
            attn[i, :L] = True

        y = torch.tensor(targets, dtype=torch.long)
        return x, attn, y


# -----------------------------
# Model: small causal Transformer
# -----------------------------
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
        # x: (B,T), attn_mask: (B,T) True for real tokens
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        h = self.tok(x) + self.pos(pos_ids)

        # zero padded positions (stability on some CUDA paths)
        h = h.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        # boolean causal mask: True means blocked
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        # padding mask: True means ignore key
        key_pad = ~attn_mask

        h = self.enc(h, mask=causal, src_key_padding_mask=key_pad)

        # re-zero padded outputs
        h = h.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        lengths = attn_mask.long().sum(dim=1)
        if (lengths <= 0).any():
            raise RuntimeError("Empty (all-pad) sequence encountered.")
        last_idx = lengths - 1

        last_h = h[torch.arange(B, device=x.device), last_idx]
        logits = self.head(last_h)
        return logits


# -----------------------------
# Evaluation: Recall@k on edges
# -----------------------------
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
            # RIGHT padding
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


# -----------------------------
# Training with early stopping on val R@10(subset)
# -----------------------------
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
    epochs: int = 50
    seed: int = 42
    val_ratio: float = 0.1
    patience: int = 8
    min_delta: float = 1e-4


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transformer(train_seqs, val_seqs, K: int, cfg: TrainCfg, device: str):
    # clusters are [0..K-1]
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
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing if hasattr(cfg, 'label_smoothing') else 0.0)

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
                print("Loss is NaN/Inf. Debug batch:")
                print("x[0]:", x[0].detach().cpu().tolist())
                print("attn[0]:", attn[0].detach().cpu().tolist())
                print("y[0]:", y[0].item())
                raise RuntimeError("Non-finite loss")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total_loss / max(1, n)

        # Validation: focus on subset (history exists) and R@10
        val_scores, val_edges = eval_model_on_edges(
            model, val_seqs, device, cfg.max_ctx, pad_id, bos_id, ks=ks, trigram_subset=True
        )
        val_r10 = val_scores[10]

        print(f"[Epoch {ep}/{cfg.epochs}] train_loss={train_loss:.4f} | val_edges={val_edges} | val R@10(subset)={val_r10:.4f}")

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

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
        print(f"[Restore] Loaded best checkpoint with val R@10(subset)={best_val:.4f}")
    else:
        print("[Warn] No best_state saved (unexpected). Using last epoch weights.")

    return model, pad_id, bos_id , best_val


# -----------------------------
# Main
# -----------------------------
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
    )

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    seqs = build_sequences(args.clusters_tsv)
    K = get_num_clusters(seqs)
    print(f"[Data] traces(len>=2)={len(seqs)} | K={K}")

    sanity_check_clusters(seqs, K)

    train, test, test_apps = split_by_app(seqs, test_ratio=args.test_ratio, seed=cfg.seed)
    train_tr, val, val_apps = split_train_val_by_app(train, val_ratio=cfg.val_ratio, seed=cfg.seed)

    print(f"[Split] train_traces={len(train)} test_traces={len(test)} test_apps={len(test_apps)}")
    print(f"[Val]   train_inner={len(train_tr)} val_inner={len(val)} val_apps={len(val_apps)}")

    model, pad_id, bos_id , best_val = train_transformer(train_tr, val, K, cfg, device)

    ks = (1, 3, 5, 10)
    sc_all, e_all = eval_model_on_edges(model, test, device, cfg.max_ctx, pad_id, bos_id, ks=ks, trigram_subset=False)
    sc_sub, e_sub = eval_model_on_edges(model, test, device, cfg.max_ctx, pad_id, bos_id, ks=ks, trigram_subset=True)

    print("\n=== Transformer next-step prediction (BEST checkpoint) ===")
    print(f"Edges evaluated (all bigram edges): {e_all}")
    print(f"Edges evaluated (trigram-subset):   {e_sub}")
    print(" " * 18 + "R@1     R@3     R@5     R@10")
    print("Transformer (all)     " + "  ".join([f"{sc_all[k]:.4f}" for k in ks]))
    print("Transformer (subset)  " + "  ".join([f"{sc_sub[k]:.4f}" for k in ks]))

    # save model
    out = {
        "state_dict": model.state_dict(),
        "K": K,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "cfg": cfg.__dict__,
        "val_best_metric": "R@10(subset)",
    }

    best_val_str = f"experiments_weights/best_val_{best_val:.4f}"
    args.out = f"{best_val_str}_{args.out}"
    
    print(f"BEST_VAL_LOSS: {best_val:.4f}")
    torch.save(out, args.out)
    print(f"\nSaved: {args.out}")


def create_out_file(args, ap):
    """
    defaults = {
    action.dest: action.default
    for action in ap._actions
    if action.dest != "help"
    }

    passed_args = []

    for k, v in vars(args).items():
        if (
            k in defaults
            and v != defaults[k]
            and isinstance(v, (int, float))
        ):
            passed_args.append(str(v))
    
    if args.out is None:
        args.out = "transformer_next_cluster_" + "_".join(passed_args) + ".pt"
    
    print(f"[Output] {args.out}")
    
    """

    argv = sys.argv[1:]
    explicit_dests = set()

    for action in ap._actions:
        if action.dest == "help":
            continue
        # If any of the option strings (e.g. "--epochs") appear in argv, mark it explicit
        if any(opt in argv for opt in action.option_strings):
            explicit_dests.add(action.dest)

    passed_args = []
    for k, v in vars(args).items():
        if k in explicit_dests and isinstance(v, (int, float)):
            passed_args.append(str(v))

    if args.out is None:
        args.out = "transformer_next_cluster_" + "_".join(passed_args) + ".pt"
    
    
    print(f"[Output] {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clusters_tsv", help="path to screen_clusters_kXX.tsv")

    # core
    ap.add_argument("--ctx", type=int, default=16, help="max context length (prefix length)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    #ap.add_argument("--out", type=str, default="transformer_next_cluster.pt")
    ap.add_argument("--out", type=str, default=None)
    
    # model size
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    # optimization
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

     
    args = ap.parse_args()
    
    create_out_file(args, ap)
    
    #print(f"[Output] {args.out}")

    main(args)


"""
python 13_2_transformer_next_cluster.py processed_data/clusters/screen_clusters_k40.tsv --ctx 16 --epochs 200 --patience 8 --val_ratio 0.1

python 13_2_transformer_next_cluster.py processed_data/clusters/screen_clusters_k120.tsv --ctx 2  --epochs 200 --patience 8
python 13_2_transformer_next_cluster.py processed_data/clusters/screen_clusters_k120.tsv --ctx 8  --epochs 200 --patience 8
python 13_2_transformer_next_cluster.py processed_data/clusters/screen_clusters_k120.tsv --ctx 16 --epochs 200 --patience 8


### CHANGING SOME HYPERPARAMS

python 13_2_transformer_next_cluster.py processed_data/clusters/screen_clusters_k40.tsv --ctx 16 --epochs 200 --patience 8 --val_ratio 0.1 --lr 1e-4 --dropout 0.2 


python 13_2_transformer_next_cluster.py processed_data/clusters/screen_clusters_k40.tsv --ctx 16 --epochs 200 --patience 8 --val_ratio 0.1 --lr 1e-4 --dropout 0.2 --label_smoothing 0.1

python 13_2_transformer_next_cluster.py processed_data/clusters/screen_clusters_k40.tsv --ctx 8 --epochs 200 --patience 8 --val_ratio 0.1 --lr 1e-4 --dropout 0.2 --label_smoothing 0.1

"""

