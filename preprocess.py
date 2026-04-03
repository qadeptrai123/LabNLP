"""
Preprocessing pipeline for SemEval Task A.
Extracts stylometric (agnostic) features and saves them to parquet files.

Usage:
    python preprocess.py
    python preprocess.py --data-dir ./Task_A_Processed   # override output dir
    python preprocess.py --limit 1000                    # process only N rows (debug)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress unsafe torch checkpoint warnings
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *a, **kw: True
transformers.modeling_utils.check_torch_load_is_safe    = lambda *a, **kw: True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.config    import DEFAULT_CONFIG
from src.features  import AgnosticFeatureExtractor


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="SemEval Task A — Feature Preprocessing")
    parser.add_argument("--data-dir",  default=None,  help="Override processed data directory")
    parser.add_argument("--limit",     type=int, default=None, help="Process only N rows (debug)")
    parser.add_argument("--no-ppl",   action="store_true", help="Skip perplexity extraction (faster)")
    return parser.parse_args()


# ── Pipeline ─────────────────────────────────────────────────────────────────
def process_split(extractor, input_path: str, output_path: str, limit: int = None):
    """Extract features from one parquet split and save."""
    if not os.path.exists(input_path):
        print(f"[WARN] Not found: {input_path}  — skipping.")
        return

    df = pd.read_parquet(input_path)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
        print(f"  Limited to {limit} rows.")

    print(f"  Extracting features for {len(df)} samples...")
    features_list = []
    for code in tqdm(df['code'], desc="  Processing", dynamic_ncols=True):
        try:
            feats = extractor.extract_all(code)
        except Exception as e:
            print(f"\n  Error on code snippet: {e}  — using zero-vector.")
            feats = [0.0] * 11
        features_list.append(feats)

    df['agnostic_features'] = features_list
    df.to_parquet(output_path)

    arr = np.array(features_list)
    print(f"  Saved {len(df)} rows -> {output_path}")
    print(f"  Feature matrix: {arr.shape}  |  Avg perplexity: {np.nanmean(arr[:, 0]):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Build config (CLI overrides)
    config = DEFAULT_CONFIG.copy()
    config["data"] = {**DEFAULT_CONFIG["data"]}
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir

    raw_dir  = config["data"]["raw_data_dir"]
    proc_dir = config["data"]["data_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    # Paths
    train_in  = os.path.join(raw_dir,  "train.parquet")
    val_in   = os.path.join(raw_dir,  "validation.parquet")
    train_out = os.path.join(proc_dir, "train_processed.parquet")
    val_out   = os.path.join(proc_dir, "val_processed.parquet")

    # Suppress safe-check overrides already applied above
    if args.no_ppl:
        print("[INFO] Skipping perplexity — using zero for all rows.")
        # Walk-around: stub out the perplexity method
        def _stub(self, code):
            return 0.0
        AgnosticFeatureExtractor._compute_perplexity = _stub

    # Device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Config: project={config['common']['project_name']}  "
          f"ppl_model={config['data']['perplexity_model']}  "
          f"out_dir={proc_dir}")

    # Init extractor — load from local model dir, not HuggingFace
    model_path = config["data"]["perplexity_model_path"]
    extractor = AgnosticFeatureExtractor(model_path, device)

    # Process
    print(f"\n--- TRAIN ---")
    process_split(extractor, train_in, train_out, args.limit)

    print(f"\n--- VAL ---")
    process_split(extractor, val_in, val_out, args.limit)

    print("\nDone.")


if __name__ == "__main__":
    main()
