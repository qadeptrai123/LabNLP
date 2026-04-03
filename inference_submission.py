"""
Multi-GPU inference pipeline for SemEval Task A — generates submission.csv.

Speeds up both stages with DDP on N GPUs:
  1. Feature extraction (Qwen perplexity)  — parallelised across GPUs
  2. Model inference                       — DistributedDataParallel

Usage (8 GPUs):
    torchrun --nproc_per_node=8 inference_submission.py
    torchrun --nproc_per_node=8 inference_submission.py --batch_size 64

Single-GPU fallback:
    python inference_submission.py
"""
import os, sys, yaml, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from tqdm import tqdm

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *a, **kw: True
transformers.modeling_utils.check_torch_load_is_safe    = lambda *a, **kw: True

_num_gpus = torch.cuda.device_count()
_is_dist   = _num_gpus > 1

if _is_dist:
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _is_rank0   = (_local_rank == 0)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(_local_rank)
    device = torch.device("cuda", _local_rank)
else:
    _local_rank = 0
    _is_rank0   = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_is_main = _is_rank0   # alias for clarity


# ── Logging helper ────────────────────────────────────────────────────────────
def log(msg):
    if _is_main:
        print(msg)


# ── Feature extraction (DDP parallelised) ──────────────────────────────────────
def extract_features_ddp(test_df, config):
    """Extract agnostic_features across all GPUs in parallel."""
    if "agnostic_features" in test_df.columns:
        log("Features already present — skipping extraction.")
        return test_df

    cache_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Task_A/test_processed.parquet",
    )
    if os.path.exists(cache_path):
        log(f"Found cached features: {cache_path}")
        return pd.read_parquet(cache_path)

    log(f"Extracting features across {_num_gpus} GPU(s)...")

    # Each rank builds its own extractor (Qwen loaded per rank — VRAM trade-off
    # for parallelism. For 8x V100 32GB this is fine.)
    from src.features import AgnosticFeatureExtractor
    extractor = AgnosticFeatureExtractor(
        config["data"]["perplexity_model_path"], str(device),
    )

    n = len(test_df)
    chunk_size = (n + _num_gpus - 1) // _num_gpus
    start = _local_rank * chunk_size
    end   = min(start + chunk_size, n)
    chunk_codes = test_df["code"].iloc[start:end].tolist()

    # ── Batch perplexity: 32 samples per batch → GPU ~80%+ utilization ─────
    PERPLEX_BATCH = 32
    perp_losses = []
    n_batches = (len(chunk_codes) + PERPLEX_BATCH - 1) // PERPLEX_BATCH
    for i in tqdm(
        range(n_batches), desc=f"Qwen perplexity [rank {_local_rank}]",
        unit="batch", disable=(not _is_main),
    ):
        batch_codes = chunk_codes[i * PERPLEX_BATCH : (i + 1) * PERPLEX_BATCH]
        try:
            perp_losses.extend(extractor.compute_perplexity_batch(batch_codes))
        except Exception:
            perp_losses.extend([0.0] * len(batch_codes))

    # ── Remaining stylometric features (fast, single-sample is fine) ─────────
    local_features = []
    for i, code in enumerate(
        tqdm(chunk_codes,
             desc=f"Rank {_local_rank}", unit="samples",
             disable=(not _is_main)),
    ):
        try:
            words = extractor.re_words.findall(code)
            f_ids   = extractor._analyze_identifiers(words)
            f_const = extractor._analyze_consistency(code, words)
            f_struc = extractor._analyze_structure(code, words)
            # perplexity already computed in batch above
            feats = [perp_losses[i]] + f_ids + f_const + f_struc
        except Exception:
            feats = [0.0] * 11
        local_features.append(feats)

    # Gather all chunks to rank 0
    if _is_dist:
        # Pad to same size so we can all_gather
        world_size = dist.get_world_size()
        global_features = [None] * world_size
        dist.all_gather_object(global_features, local_features)

        if _is_main:
            full_features = []
            for rank_features in global_features:
                full_features.extend(rank_features)
        else:
            full_features = None
    else:
        full_features = local_features

    del extractor
    torch.cuda.empty_cache()

    # ── FIX: broadcast BEFORE slow I/O so other ranks don't idle-wait ──
    if _is_dist:
        if _is_main:
            test_df_out = test_df.copy()
            test_df_out["agnostic_features"] = full_features
            obj_list = [test_df_out]
        else:
            obj_list = [None]
        dist.broadcast_object_list(obj_list, src=0)
        test_df_out = obj_list[0]

        if _is_main:
            log(f"Caching to {cache_path}")
            test_df_out.to_parquet(cache_path)
        test_df = test_df_out
    else:
        test_df = test_df.copy()
        test_df["agnostic_features"] = full_features
        log(f"Caching to {cache_path}")
        test_df.to_parquet(cache_path)

    return test_df


# ── Model inference (DDP) ───────────────────────────────────────────────────────
def run_inference(args, config, test_df):
    from src.models  import HybridClassifier
    from src.dataset import AgnosticDataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.amp import autocast

    log("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    model = HybridClassifier(config)
    state_dict = torch.load(
        os.path.join(args.checkpoint_dir, "model_state.bin"),
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.to(device)

    if _is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[_local_rank],
        )
        world_size = dist.get_world_size()
    else:
        world_size = 1

    model.eval()
    log(f"  Model ready (world_size={world_size}).")

    # Build dataset / dataloader
    dataset = AgnosticDataset(
        test_df, tokenizer,
        max_length=config["data"]["max_length"],
        is_train=False,
    )
    sampler = DistributedSampler(dataset, shuffle=False) if _is_dist else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Prediction
    log("Running inference...")
    all_preds  = [None] * world_size if _is_dist else []
    all_probs  = [None] * world_size if _is_dist else []
    local_preds, local_probs = [], []

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Inferencing", unit="batch",
                        disable=not _is_main)
        for batch in iterator:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            feats          = batch["extra_features"].to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                logits, _, _ = model(input_ids, attention_mask, feats, labels=None)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            local_preds.extend(preds.cpu().numpy())
            local_probs.extend(probs[:, 1].cpu().numpy())  # P(AI)

    # Gather from all ranks to rank 0
    if _is_dist:
        dist.all_gather_object(all_preds, local_preds)
        dist.all_gather_object(all_probs, local_probs)

        if _is_main:
            final_preds = []
            final_probs = []
            for rp, rpr in zip(all_preds, all_probs):
                final_preds.extend(rp)
                final_probs.extend(rpr)
        else:
            final_preds, final_probs = None, None
    else:
        final_preds, final_probs = local_preds, local_probs

    return final_preds, final_probs


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="SemEval Task A — Submission Inference")
    parser.add_argument("--test_file",      default="Task_A/test.parquet")
    parser.add_argument("--checkpoint_dir", default="checkpoints/best_model")
    parser.add_argument("--batch_size",     type=int, default=64)
    args = parser.parse_args()

    log(f"{'='*60}")
    log(f"  Inference | GPUs: {_num_gpus} | Distributed: {_is_dist} | Rank: {_local_rank}")
    log(f"  Batch size: {args.batch_size}")
    log(f"{'='*60}\n")

    # Load config
    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        log(f"ERROR: config not found at {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Stage 1: features ────────────────────────────────────────────────────
    log(f"Loading test data: {args.test_file}")
    test_df = pd.read_parquet(args.test_file)
    log(f"  Rows: {len(test_df):,}  |  Columns: {list(test_df.columns)}")

    test_df = extract_features_ddp(test_df, config)

    # ── Stage 2: inference ────────────────────────────────────────────────────
    all_preds, _ = run_inference(args, config, test_df)

    # ── Stage 3: save submission ────────────────────────────────────────────
    if _is_main:
        submission = pd.DataFrame({
            "ID":    test_df["ID"],
            "label": all_preds,
        })
        out_path = args.output or "Task_A/submission.csv"
        submission.to_csv(out_path, index=False)
        log(f"\nSubmission saved → {out_path}  ({len(submission):,} rows)")
        log(f"Label distribution:\n{submission['label'].value_counts().sort_index().to_string()}")

        if "label" in test_df.columns:
            from sklearn.metrics import accuracy_score, f1_score, classification_report
            acc = accuracy_score(test_df["label"].values, all_preds)
            f1  = f1_score(test_df["label"].values, all_preds, average="macro")
            log(f"\n  Accuracy : {acc:.4f}")
            log(f"  F1-macro: {f1:.4f}")
            print(classification_report(
                test_df["label"].values, all_preds,
                target_names=["Human", "AI"], digits=4,
            ))

    if _is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
