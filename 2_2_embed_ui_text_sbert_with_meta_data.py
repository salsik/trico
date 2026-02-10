import os, json, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def iter_view_jsons(root):
    return sorted(glob.glob(os.path.join(root, "*", "*", "view_hierarchies", "*.json")))

def extract_ids_from_view_json_path(p):
    # root/app_id/trace_id/view_hierarchies/file.json
    parts = os.path.normpath(p).split(os.sep)
    vh = parts.index("view_hierarchies")
    app_id = parts[vh-2]
    trace_id = parts[vh-1]
    screen_id = os.path.splitext(os.path.basename(p))[0]
    return app_id, trace_id, screen_id

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
    out, seen = [], set()
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
        for k in ["text","content-desc","contentDescription","hint","label"]:
            if isinstance(n.get(k), str) and n[k].strip():
                text = n[k].strip()
                break
        key = (cls, text)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{cls}: {text}" if text else cls)
        if len(out) >= max_elems:
            break
    return "UI_SCREEN\n" + "\n".join(out)

def main(root, out_npy="sbert_text_embeddings.npy", out_tsv="sbert_text_meta.tsv",
         model_name="all-MiniLM-L6-v2", batch_size=128):
    paths = iter_view_jsons(root)
    rows, texts = [], []
    for p in tqdm(paths, desc="Parsing view_hierarchies"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        view = data.get("activity", data.get("view", data)) if isinstance(data, dict) else data
        ser = serialize_screen_text(view)
        if len(ser.strip()) < 10:
            continue
        app_id, trace_id, screen_id = extract_ids_from_view_json_path(p)
        screen_key = f"{app_id}::{trace_id}::{screen_id}"
        rows.append((screen_key, app_id, trace_id, screen_id, p))
        texts.append(ser)

    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    np.save(out_npy, embs.astype(np.float32))

    df = pd.DataFrame(rows, columns=["screen_key","app_id","trace_id","screen_id","json_path"])
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved {out_npy} with shape {embs.shape}")
    print(f"Saved {out_tsv} with {len(df)} rows")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    main(args.root)
