# 12_transformer_next_cluster.py
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Reuse the same preprocessing style as your baselines
# -----------------------------
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

        # collapse consecutive duplicates (same as your scripts)
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

# -----------------------------
# Dataset: prefix -> next token
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
            # RIGHT padding: put tokens at the start
            x[i, :L] = torch.tensor(p, dtype=torch.long)
            attn[i, :L] = True

        y = torch.tensor(targets, dtype=torch.long)
        return x, attn, y


    def collate1(self, batch):
        prefixes, targets = zip(*batch)
        B = len(prefixes)
        T = min(self.max_ctx, max(len(p) for p in prefixes))
        x = torch.full((B, T), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.bool)

        for i, p in enumerate(prefixes):
            p = p[-T:]
            x[i, -len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, -len(p):] = True

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

        # prevent padded positions from contaminating attention/residuals
        h = h.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        # boolean causal mask: True means "blocked"
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        # padding mask: True means "ignore key"
        key_pad = ~attn_mask

        h = self.enc(h, mask=causal, src_key_padding_mask=key_pad)

        # zero out padded outputs again (important on some CUDA paths)
        h = h.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        lengths = attn_mask.long().sum(dim=1)
        if (lengths <= 0).any():
            raise RuntimeError("Empty (all-pad) sequence encountered.")
        last_idx = lengths - 1

        last_h = h[torch.arange(B, device=x.device), last_idx]

        # IMPORTANT: use the correct output layer name
        logits = self.head(last_h)  # or self.lm_head if that's what you defined
        return logits

    def forward1(self, x, attn_mask):
        # x: (B,T), attn_mask: (B,T) True for real tokens
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        h = self.tok(x) + self.pos(pos_ids)

        # IMPORTANT: force padded positions to be 0 (prevents NaN propagation)
        h = h.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        # causal mask as FLOAT (more stable across versions)
        causal = torch.full((T, T), float("-inf"), device=x.device)
        causal = torch.triu(causal, diagonal=1)  # -inf above diagonal, 0 elsewhere
        # note: TransformerEncoder expects additive mask where -inf blocks attention

        key_pad = ~attn_mask  # True means "ignore key"
        h = self.enc(h, mask=causal, src_key_padding_mask=key_pad)

        # re-zero padded outputs (belt-and-suspenders)
        h = h.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        last_idx = attn_mask.long().sum(dim=1) - 1
        last_h = h[torch.arange(B, device=x.device), last_idx]
        logits = self.head(last_h)
        #logits = self.lm_head(last_h)
        return logits

    
    
    def forward_old(self, x, attn_mask):
        # x: (B,T), attn_mask: (B,T) True for real tokens
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(pos_ids)

        # causal mask (block attention to future positions)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        # src_key_padding_mask: True means "ignore"
        key_pad = ~attn_mask
        h = self.enc(h, mask=causal, src_key_padding_mask=key_pad)

        # take last valid token representation per row
        last_idx = attn_mask.long().sum(dim=1) - 1  # (B,)
        last_h = h[torch.arange(B, device=x.device), last_idx]
        logits = self.head(last_h)  # (B, V)
        return logits

# -----------------------------
# Metrics: Recall@k
# -----------------------------
@torch.no_grad()
def recall_at_k_from_logits(logits, y, ks=(1,3,5,10)):
    ranked = torch.argsort(logits, dim=1, descending=True)
    out = {}
    for k in ks:
        hit = (ranked[:, :k] == y.unsqueeze(1)).any(dim=1).float().mean().item()
        out[k] = hit
    return out

@torch.no_grad()
def eval_model_on_edges(model, test_seqs, device, max_ctx, pad_id, bos_id, ks=(1,3,5,10), trigram_subset=False):
    """
    Evaluate next-step recall.
    - trigram_subset=False: evaluate on all bigram edges (t>=1)
    - trigram_subset=True : evaluate only where history exists (t>=2) to mimic trigram-edge subset
    """
    model.eval()
    hits = {k: 0 for k in ks}
    total = 0

    for _, _, seq in test_seqs:
        seq = [bos_id] + list(seq)
        # positions:
        # predict token at index t given prefix seq[:t]
        start_t = 2 if trigram_subset else 1
        for t in range(start_t, len(seq)):
            prefix = seq[:t]
            target = seq[t]

            # build a single-item batch
            T = min(max_ctx, len(prefix))
            x = torch.full((1, T), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((1, T), dtype=torch.bool, device=device)
            
            p = prefix[-T:]
            ## changed to right padding
            #x[0, -len(p):] = torch.tensor(p, dtype=torch.long, device=device)
            #attn[0, -len(p):] = True

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

# -----------------------------
# Train
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
    epochs: int = 5
    seed: int = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_transformer(train_seqs, K, cfg: TrainCfg, device):
    # Reserve special tokens at end:
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
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, cfg.epochs + 1):
        total_loss = 0.0
        n = 0
        for x, attn, y in dl:
            x, attn, y = x.to(device), attn.to(device), y.to(device)
            logits = model(x, attn)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        if not torch.isfinite(loss):
            print("Loss is NaN/Inf. Debug batch:")
            print("x[0]:", x[0].detach().cpu().tolist())
            print("attn[0]:", attn[0].detach().cpu().tolist())
            print("y[0]:", y[0].item())
            raise RuntimeError("Non-finite loss")
        
        print(f"[Epoch {ep}/{cfg.epochs}] loss={total_loss/max(1,n):.4f}")

    return model, pad_id, bos_id

def main(
    clusters_tsv,
    test_ratio_apps=0.2,
    seed=42,
    max_ctx=16,
    epochs=5,
):
    cfg = TrainCfg(max_ctx=max_ctx, epochs=epochs, seed=seed)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    seqs = build_sequences(clusters_tsv)
    K = get_num_clusters(seqs)
    print(f"[Data] traces(len>=2)={len(seqs)} | K={K}")

    
    # Sanity: cluster ids must be >=0
    all_ids = []
    for _, _, seq in seqs:
        all_ids.extend(seq)
    mn, mx = min(all_ids), max(all_ids)
    print(f"[Sanity] cluster_id min={mn} max={mx} K={K}")
    assert mn >= 0, "Found negative cluster_id (e.g., -1). This will break CrossEntropyLoss."
    assert mx < K, "Found cluster_id >= K (shouldn't happen if K=max+1)."

        
    
    train, test, test_apps = split_by_app(seqs, test_ratio=test_ratio_apps, seed=seed)
    print(f"[Split] train_traces={len(train)} test_traces={len(test)} test_apps={len(test_apps)}")

    model, pad_id, bos_id = train_transformer(train, K, cfg, device)
    
    ks = (1,3,5,10)
    sc_all, e_all = eval_model_on_edges(model, test, device, cfg.max_ctx, pad_id, bos_id, ks=ks, trigram_subset=False)
    sc_sub, e_sub = eval_model_on_edges(model, test, device, cfg.max_ctx, pad_id, bos_id, ks=ks, trigram_subset=True)

    print("\n=== Transformer next-step prediction ===")
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
    }
    torch.save(out, "transformer_next_cluster.pt")
    print("\nSaved: transformer_next_cluster.pt")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python 12_transformer_next_cluster.py <screen_clusters_kXX.tsv> [max_ctx] [epochs]")
        sys.exit(1)
    clusters_tsv = sys.argv[1]
    max_ctx = int(sys.argv[2]) if len(sys.argv) >= 3 else 16
    epochs = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    main(clusters_tsv, max_ctx=max_ctx, epochs=epochs)




"""""
python 13_transformer_experiment.py processed_data/clusters/screen_clusters_k40.tsv 16 5
python 13_transformer_experiment.py processed_data/clusters/screen_clusters_k120.tsv 16 5

"""