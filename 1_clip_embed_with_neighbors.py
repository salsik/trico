import os
import glob
import random
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import open_clip


IMG_EXTS = ("png", "jpg", "jpeg", "webp")


def iter_images(root: str, exts=IMG_EXTS) -> List[str]:
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, "**", f"*.{e}"), recursive=True)
    return sorted(files)


def extract_app_trace_ids(image_path: str) -> Tuple[str, str]:
    """
    Attempts to infer (app_id, trace_id) from a screenshot path.

    Primary assumption:
      .../<app_id>/<trace_id>/screenshots/<img>

    If 'screenshots' isn't present, fallback to:
      .../<app_id>/<trace_id>/<img> (two parents up)

    If still ambiguous, returns "unknown_app"/"unknown_trace".
    """
    parts = os.path.normpath(image_path).split(os.sep)

    # Try to locate the 'screenshots' directory in the path
    if "screenshots" in parts:
        idx = len(parts) - 1 - parts[::-1].index("screenshots")  # last occurrence
        # Expect: ... / app_id / trace_id / screenshots / file
        if idx >= 2:
            trace_id = parts[idx - 1]
            app_id = parts[idx - 2]
            return app_id, trace_id

    # Fallback: take two directory levels up from the file
    if len(parts) >= 3:
        trace_id = parts[-2]
        app_id = parts[-3]
        return app_id, trace_id

    return "unknown_app", "unknown_trace"


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


@torch.no_grad()
def embed_images_clip(
    image_paths: List[str],
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    all_embs = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            im = Image.open(p).convert("RGB")
            imgs.append(preprocess(im))
        x = torch.stack(imgs).to(device)

        feats = model.encode_image(x)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)

        all_embs.append(feats.cpu().numpy().astype(np.float32))

    embs = np.concatenate(all_embs, axis=0)
    return embs


def save_tsv_with_ids(
    out_tsv: str,
    image_paths: List[str],
    embs: np.ndarray,
) -> None:
    # Columns: app_id, trace_id, image_path, e0..eD-1
    with open(out_tsv, "w", encoding="utf-8") as f:
        header = ["app_id", "trace_id", "image_path"] + [f"e{i}" for i in range(embs.shape[1])]
        f.write("\t".join(header) + "\n")

        for p, v in zip(image_paths, embs):
            app_id, trace_id = extract_app_trace_ids(p)
            row = [app_id, trace_id, p.replace("\t", " ")] + [f"{x:.6f}" for x in v]
            f.write("\t".join(row) + "\n")


def cosine_topk_neighbors(
    embs: np.ndarray,
    idx: int,
    k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Returns list of (neighbor_index, cosine_similarity), excluding idx itself.
    Assumes embs are already L2-normalized.
    """
    q = embs[idx]  # (D,)
    sims = embs @ q  # (N,)
    sims[idx] = -np.inf
    topk = np.argpartition(-sims, k)[:k]
    topk = topk[np.argsort(-sims[topk])]
    return [(int(j), float(sims[j])) for j in topk]


def sanity_check_neighbors(
    image_paths: List[str],
    embs: np.ndarray,
    num_queries: int = 5,
    topk: int = 5,
    seed: int = 42,
) -> None:
    """
    Prints top-k nearest neighbors for a few random query screens.
    """
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)

    # If not normalized, normalize now for cosine sim
    # (If you used normalize=True in embedding, this is redundant but safe.)
    embs = l2_normalize(embs)

    rng = random.Random(seed)
    n = len(image_paths)
    picks = [rng.randrange(n) for _ in range(min(num_queries, n))]

    print("\n=== SANITY CHECK: Top neighbors by cosine similarity ===")
    for qi in picks:
        qpath = image_paths[qi]
        q_app, q_trace = extract_app_trace_ids(qpath)

        print("\n---")
        print(f"QUERY idx={qi}  sim=1.0000")
        print(f"  app_id={q_app}  trace_id={q_trace}")
        print(f"  path={qpath}")

        nbrs = cosine_topk_neighbors(embs, qi, k=topk)
        for rank, (j, sim) in enumerate(nbrs, start=1):
            p = image_paths[j]
            app_id, trace_id = extract_app_trace_ids(p)
            print(f"  #{rank}: idx={j}  sim={sim:.4f}  app_id={app_id}  trace_id={trace_id}")
            print(f"      {p}")


def main(
    image_root: str,
    out_tsv: str = "clip_embeddings_with_ids.tsv",
    out_npy: str = "clip_embeddings.npy",
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    batch_size: int = 64,
    normalize: bool = True,
    sanity_num_queries: int = 5,
    sanity_topk: int = 5,
):
    paths = iter_images(image_root)
    if not paths:
        raise FileNotFoundError(f"No images found under: {image_root}")

    embs = embed_images_clip(
        paths,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        normalize=normalize,
    )

    np.save(out_npy, embs)
    save_tsv_with_ids(out_tsv, paths, embs)

    print(f"\nDone embedding.")
    print(f"Images: {len(paths)} | dim={embs.shape[1]}")
    print(f"Saved: {out_npy}")
    print(f"Saved: {out_tsv}")

    sanity_check_neighbors(
        image_paths=paths,
        embs=embs,
        num_queries=sanity_num_queries,
        topk=sanity_topk,
        seed=42,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("image_root", help="Root folder containing screenshots (nested OK).")
    ap.add_argument("--out_tsv", default="clip_embeddings_with_ids.tsv")
    ap.add_argument("--out_npy", default="clip_embeddings.npy")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--no_normalize", action="store_true", help="Disable L2 normalization of embeddings.")
    ap.add_argument("--sanity_queries", type=int, default=5)
    ap.add_argument("--sanity_topk", type=int, default=5)

    args = ap.parse_args()

    main(
        image_root=args.image_root,
        out_tsv=args.out_tsv,
        out_npy=args.out_npy,
        model_name=args.model,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        normalize=(not args.no_normalize),
        sanity_num_queries=args.sanity_queries,
        sanity_topk=args.sanity_topk,
    )


# python clip_embed_with_neighbors.py ../../dataset/traces/filtered_traces/  --batch_size 64 --sanity_queries 10 --sanity_topk 5