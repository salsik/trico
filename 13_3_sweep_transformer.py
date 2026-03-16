import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


BEST_VAL_LOSS_RE = re.compile(r"BEST_VAL_LOSS:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
METRICS_JSON_RE = re.compile(r"METRICS_JSON:(\{.*\})")

EPOCH_PROGRESS_RE = re.compile(r"\[Epoch\s+(\d+)\s*/\s*(\d+)\]")


def run_id_from_params(params: Dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:10]


def build_command(
    python_exe: str,
    train_script: str,
    data_path: str,
    base_args: List[str],
    params: Dict[str, Any],
) -> List[str]:
    #cmd = [python_exe, train_script, data_path] + base_args
    cmd = [python_exe, train_script] + base_args
    for k, v in params.items():
        flag = f"--{k}"
        # allow boolean flags if you add them later
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def parse_best_metric(stdout_text: str, metric_key: str) -> Optional[float]:
    """
    Supports two formats:
      1) 'BEST_VAL_LOSS: <float>' if metric_key == 'best_val_loss'
      2) 'METRICS_JSON:{...}' where JSON contains metric_key
    """
    # JSON metrics
    for m in METRICS_JSON_RE.finditer(stdout_text):
        try:
            obj = json.loads(m.group(1))
            if metric_key in obj and obj[metric_key] is not None:
                return float(obj[metric_key])
        except Exception:
            pass

    # Simple BEST_VAL_LOSS line (common)
    if metric_key == "best_val_loss":
        m = BEST_VAL_LOSS_RE.search(stdout_text)
        if m:
            return float(m.group(1))

    return None


def all_combinations(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    combos = []
    for prod in itertools.product(*vals):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def sampled_combinations(space: Dict[str, List[Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(space.keys())
    combos = []
    for _ in range(n):
        params = {k: rng.choice(space[k]) for k in keys}
        combos.append(params)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in combos:
        s = json.dumps(p, sort_keys=True)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def parse_last_trained_epoch(stdout_text: str) -> Optional[int]:
    """
    Finds the last occurrence of: [Epoch X/Y]
    Returns X as int, or None if not found.
    """
    matches = EPOCH_PROGRESS_RE.findall(stdout_text)
    if not matches:
        return None
    last_epoch_str, _total_str = matches[-1]
    return int(last_epoch_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--train_script", default="13_2_transformer_next_cluster.py")
    ap.add_argument("--data_path", default="processed_data/clusters/screen_clusters_k80.tsv")
    ap.add_argument("--out_dir", default="sweeps/transformer")
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--num_trials", type=int, default=30, help="Used only for random mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", default="best_val_loss", help="Metric key to parse (lower is better by default)")
    ap.add_argument("--maximize",default=True, action="store_true", help="If set, higher metric is better")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Base args you always pass (keep these stable) ----
    base_args = [
        # Example base args (edit as you like)
        # "--some_fixed_flag", "value",
    ]

    # ---- Hyperparameter search space (EDIT THESE LISTS) ----
    # Include only flags that your training script actually supports.
    space = {
        "ctx": [ 8, 16, 24], # [ 8, 16, 32],
        "lr": [3e-5], #, 1e-4, 3e-4],
        "val_ratio": [0.1, 0.2], #  [0.05, 0.1, 0.2]
        "dropout": [ 0.2, 0.3], #[0.1, 0.2, 0.3],
        "label_smoothing": [0.0], # , 0.05, 0.1],
        "epochs": [200],          # keep fixed or vary
        "patience": [8],      # early stopping sensitivity
    }

    # OPTIONAL ideas (only if supported in your training script):
    space.update({
         "weight_decay": [0.0, 0.1],# [0.0, 0.01, 0.1],
    #     "warmup_steps": [0, 200, 500],
    #     "grad_clip": [0.0, 0.5, 1.0],
         "batch_size": [256, 512],
    #     "seed": [1, 2, 3],
     })
    

    space2_k_only = {
        "clusters_tsv" : [  # 20,40,80,120,200
           "processed_data/clusters/screen_clusters_k20.tsv", "processed_data/clusters/screen_clusters_k40.tsv","processed_data/clusters/screen_clusters_k80.tsv","processed_data/clusters/screen_clusters_k120.tsv","processed_data/clusters/screen_clusters_k200.tsv"],
        "ctx": [ 16], # [ 8, 16, 32],
        "lr": [3e-5], #, 1e-4, 3e-4],
        "val_ratio": [0.2], #  [0.05, 0.1, 0.2]
        "dropout": [0.3], #[0.1, 0.2, 0.3],
        "label_smoothing": [0.0], # , 0.05, 0.1],
        "epochs": [200],          # keep fixed or vary
        "patience": [8],      # early stopping sensitivity
        "weight_decay": [0.1],
        "batch_size": [256]
    }

    if args.mode == "grid":
        trials = all_combinations(space2_k_only)
    else:
        trials = sampled_combinations(space2_k_only, args.num_trials, args.seed)

    results_csv = out_dir / "results.csv"
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    best_metric = None
    best_params = None
    best_run = None

    start_idx =0  # 13 or any number to skip initial runs

    #file_exists = results_csv.exists() and results_csv.stat().st_size > 0
    file_exists = False
    mode = "a" if file_exists else "w"

    with open(results_csv, mode, newline="", encoding="utf-8") as f:
        writer = None
        
        for j, params in enumerate(trials[start_idx:], 1):
            i = start_idx + j  # 1-based-ish display
            
            run_id = run_id_from_params(params)
            run_name = f"{i:04d}_{run_id}"
            log_path = logs_dir / f"{run_name}.log"
            meta_path = logs_dir / f"{run_name}.json"

            cmd = build_command(
                python_exe=args.python,
                train_script=args.train_script,
                data_path=args.data_path,
                base_args=base_args,
                params=params,
            )
           
            print(f"\n[{i}/{len(trials)}] {run_name}")
            print("CMD:", " ".join(cmd))

            if args.dry_run:
                metric_val = None
                rc = 0
                elapsed = 0.0
                stdout_text = ""
                last_epoch = None
            else:
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                elapsed = time.time() - t0
                rc = proc.returncode
                stdout_text = (proc.stdout or "") + "\n" + (proc.stderr or "")

                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write(stdout_text)

                metric_val = parse_best_metric(stdout_text, args.metric)
                last_epoch = parse_last_trained_epoch(stdout_text)


            meta = {
                "run_name": run_name,
                "returncode": rc,
                "elapsed_sec": round(elapsed, 3),
                "params": params,
                "metric_key": args.metric,
                "metric_value": metric_val,
                "last_trained_epoch": last_epoch,
                "cmd": cmd,
                "log_path": str(log_path),
            }
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2)

            row = {
                "run_name": run_name,
                "returncode": rc,
                "elapsed_sec": round(elapsed, 3),
                "metric_key": args.metric,
                "metric_value": metric_val,
                "last_trained_epoch": last_epoch,
                **params,
                "log_path": str(log_path),
            }

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()

            # Track best
            if metric_val is not None and rc == 0:
                if best_metric is None:
                    best_metric, best_params, best_run = metric_val, params, run_name
                else:
                    better = metric_val > best_metric if args.maximize else metric_val < best_metric
                    if better:
                        best_metric, best_params, best_run = metric_val, params, run_name
                        print("Best Metric Till now:", best_metric)

    print("\n=== SWEEP DONE ===")
    print("Results CSV:", results_csv)
    if best_run is None:
        print("No successful run produced a parseable metric.")
        print("Tip: print 'BEST_VAL_LOSS: <float>' or 'METRICS_JSON:{...}' from your training script.")
    else:
        print("BEST RUN:", best_run)
        print("BEST METRIC:", best_metric, f"({args.metric})")
        print("BEST PARAMS:", json.dumps(best_params, indent=2))


if __name__ == "__main__":
    main()

## this is to run the sweep for all config combinations

#it runs the following file with different hyperparameters
# 13_2_transformer_next_cluster

## still need to play with other hyperparameters
#python 13_3_sweep_transformer.py --mode grid


