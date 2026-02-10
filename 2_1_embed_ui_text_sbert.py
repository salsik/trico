import os
import json
import glob
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



def extract_app_trace_ids(path: str) -> Tuple[str, str]:
    parts = os.path.normpath(path).split(os.sep)

    # Prefer view_hierarchies
    if "view_hierarchies" in parts:
        idx = len(parts) - 1 - parts[::-1].index("view_hierarchies")  # last occurrence
        # ... / app_id / trace_id / view_hierarchies / file.json
        if idx >= 2:
            app_id = parts[idx - 2]
            trace_id = parts[idx - 1]
            return app_id, trace_id

    # Fallback to screenshots (if you ever use it)
    if "screenshots" in parts:
        idx = len(parts) - 1 - parts[::-1].index("screenshots")
        if idx >= 2:
            app_id = parts[idx - 2]
            trace_id = parts[idx - 1]
            return app_id, trace_id

    return "unknown_app", "unknown_trace"


def extract_app_trace_ids_old(path: str) -> Tuple[str, str]:
    parts = os.path.normpath(path).split(os.sep)
    if "screenshots" in parts:
        idx = len(parts) - 1 - parts[::-1].index("screenshots")
        if idx >= 2:
            return parts[idx - 2], parts[idx - 1]
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return "unknown_app", "unknown_trace"




def iter_view_hierarchy_jsons(root: str) -> List[str]:
    # Only JSON files inside view_hierarchies folders
    return sorted(glob.glob(os.path.join(root, "**", "view_hierarchies", "*.json"), recursive=True))



def iter_json_files_old(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))


def collect_nodes(obj):
    nodes = []
    if isinstance(obj, dict):
        nodes.append(obj)
        for v in obj.values():
            nodes.extend(collect_nodes(v))
    elif isinstance(obj, list):
        for x in obj:
            nodes.extend(collect_nodes(x))
    return nodes


def serialize_screen_text(view_json, max_elems=200):
    nodes = collect_nodes(view_json)
    out = []
    seen = set()

    for n in nodes:
        if not isinstance(n, dict):
            continue

        cls = n.get("class") or n.get("className") or n.get("type")
        if not isinstance(cls, str):
            continue
        cls = cls.split(".")[-1]

        if n.get("visible") is False:
            continue

        text = ""
        for k in ["text", "content-desc", "contentDescription", "hint", "label"]:
            if isinstance(n.get(k), str) and n[k].strip():
                text = n[k].strip()
                break

        key = (cls, text)
        if key in seen:
            continue
        seen.add(key)

        if text:
            out.append(f"{cls}: {text}")
        else:
            out.append(cls)

        if len(out) >= max_elems:
            break

    return "UI_SCREEN\n" + "\n".join(out)


def main(
    root: str,
    out_tsv: str = "sbert_text_embeddings.tsv",
    out_npy: str = "sbert_text_embeddings.npy",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
):
    model = SentenceTransformer(model_name)

    rows = []
    texts = []

    #json_paths = iter_json_files(root)
    json_paths = iter_view_hierarchy_jsons(root)

    for jp in tqdm(json_paths, desc="Parsing view hierarchies"):
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # RICO-style roots vary; keep this robust
        view = data.get("activity", data.get("view", data)) if isinstance(data, dict) else data

        text = serialize_screen_text(view)
        if len(text.strip()) < 10:
            continue

        app_id, trace_id = extract_app_trace_ids(jp)
        screen_id = os.path.splitext(os.path.basename(jp))[0]

        rows.append((app_id, trace_id, screen_id, jp, text))
        texts.append(text)
        


    print(f"Embedding {len(texts)} screens using SBERT ({model_name})")

    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(out_npy, embs)

    with open(out_tsv, "w", encoding="utf-8") as f:
        header = ["app_id", "trace_id", "screen_id", "json_path", "text"] + [f"e{i}" for i in range(embs.shape[1])]
        f.write("\t".join(header) + "\n")

        for (app_id, trace_id, screen_id, jp, text), v in zip(rows, embs):
            f.write(
                "\t".join(
                    [app_id, trace_id, screen_id, jp.replace("\t", " "), text.replace("\t", " ").replace("\n", "\\n")]
                    + [f"{x:.6f}" for x in v]
                )
                + "\n"
            )

    print(f"Saved: {out_npy} ({embs.shape})")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    main(root=args.root, model_name=args.model, batch_size=args.batch_size)


##  python embed_ui_text_sbert.py ../../dataset/traces/test_traces/

##  python embed_ui_text_sbert.py ../../dataset/traces/filtered_traces/

"""
SBERT implementation (drop-in replacement)
Install
pip install sentence-transformers tqdm numpy

Choose a model (important)

Good defaults:

all-MiniLM-L6-v2

384 dims

very fast
excellent for clustering

If you want stronger but heavier:
all-mpnet-base-v2 (768 dims)

I recommend starting with MiniLM.

Script: SBERT text embeddings with app_id / trace_id

This assumes you already have the same serialized screen text we used before.

embed_ui_text_sbert.py

"""