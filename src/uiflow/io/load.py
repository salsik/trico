from __future__ import annotations
import numpy as np
import pandas as pd

def ensure_cols(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}. Have: {list(df.columns)}")

def build_screen_key(df: pd.DataFrame) -> pd.Series:
    return df["app_id"].astype(str) + "::" + df["trace_id"].astype(str) + "::" + df["screen_id"].astype(str)

def read_meta_tsv(path: str, low_memory: bool = False) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=low_memory)

def read_embeddings_npy(path: str) -> np.ndarray:
    x = np.load(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x

def attach_screen_key(df: pd.DataFrame) -> pd.DataFrame:
    ensure_cols(df, ["app_id", "trace_id", "screen_id"], "meta")
    df = df.copy()
    df["screen_key"] = build_screen_key(df)
    return df

def align_by_key(
    meta_a: pd.DataFrame,
    emb_a: np.ndarray,
    meta_b: pd.DataFrame,
    emb_b: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns intersection-aligned (meta, emb_a_aligned, emb_b_aligned) in the order of meta_a.
    Assumes meta_a/meta_b have screen_key.
    """
    if len(meta_a) != emb_a.shape[0]:
        raise ValueError("meta_a rows != emb_a rows")
    if len(meta_b) != emb_b.shape[0]:
        raise ValueError("meta_b rows != emb_b rows")

    keys_a = meta_a["screen_key"].astype(str)
    keys_b = meta_b["screen_key"].astype(str)
    common = set(keys_a) & set(keys_b)

    a_mask = keys_a.isin(common).values
    b_mask = keys_b.isin(common).values

    meta_a2 = meta_a.loc[a_mask].copy().reset_index(drop=True)
    emb_a2 = emb_a[a_mask]

    meta_b2 = meta_b.loc[b_mask].copy().reset_index(drop=True)
    emb_b2 = emb_b[b_mask]

    # reorder B to match A using screen_key
    pos = {k: i for i, k in enumerate(meta_a2["screen_key"].tolist())}
    order = [pos[k] for k in meta_b2["screen_key"].tolist()]
    perm = np.argsort(order)
    emb_b2 = emb_b2[perm]
    meta_b2 = meta_b2.iloc[perm].reset_index(drop=True)

    # final assert aligned
    if not (meta_a2["screen_key"].values == meta_b2["screen_key"].values).all():
        raise RuntimeError("Alignment failed: screen_key mismatch after reorder.")
    return meta_a2, emb_a2, emb_b2

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)
