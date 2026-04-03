"""
Training script for SemEval Task A — Human vs AI Code Detection.
Imports shared components from src/. Single-GPU / DDP multi-GPU training,
loads models from local model/ directory.

Usage:
    python train.py
    python train.py --epochs 5 --batch-size 32
    python train.py --resume ./checkpoints/best_model
    # DDP (2+ GPUs):
    torchrun --nproc_per_node=2 train.py
    torchrun --nproc_per_node=2 train.py --epochs 5 --batch-size 32
"""
import os, sys, random, yaml, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist

_num_gpus = torch.cuda.device_count()
_is_distributed = _num_gpus > 1

if _is_distributed:
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(_local_rank)
else:
    _local_rank = 0

_is_rank0 = (_local_rank == 0)   # True only on the primary process — use for logging / saves

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Suppress unsafe torch checkpoint warnings
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *a, **kw: True
transformers.modeling_utils.check_torch_load_is_safe    = lambda *a, **kw: True

from src.config   import DEFAULT_CONFIG
from src.dataset  import AgnosticDataset
from src.models   import HybridClassifier


# ── Seed ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SemEval Task A — Training")
    p.add_argument("--data-dir",    default=None,   help="Override data directory")
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--resume",     default=None,   help="Path to checkpoint to resume from")
    return p.parse_args()


# ── Checkpointing ─────────────────────────────────────────────────────────────
def save_checkpoint(model, tokenizer, path, epoch, metrics, config):
    os.makedirs(path, exist_ok=True)
    print(f"  -> Saving to {path}")
    tokenizer.save_pretrained(path)
    state = (model.module if hasattr(model, "module") else model).state_dict()
    torch.save(state, os.path.join(path, "model_state.bin"))
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    with open(os.path.join(path, "training_meta.yaml"), "w") as f:
        yaml.safe_dump({"epoch": epoch, "metrics": metrics}, f)


def load_checkpoint(model, path, device):
    state = torch.load(os.path.join(path, "model_state.bin"), map_location=device)
    (model.module if hasattr(model, "module") else model).load_state_dict(state)
    if _is_rank0:
        print(f"  -> Loaded checkpoint from {path}")


# ── Evaluation ────────────────────────────────────────────────────────────────
LABEL_NAMES = ["Human", "AI"]


def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    preds_all, labels_all, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            feats          = batch["extra_features"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            logits, _, _ = model(input_ids, attention_mask, feats, labels=None)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    # ── Fix #2: aggregate metrics across all DDP ranks ──────────────────────
    if _is_distributed:
        world_size = dist.get_world_size()

        # Gather full preds/labels to every rank before computing sklearn metrics
        preds_tensor  = torch.tensor(preds_all,  device=device)
        labels_tensor = torch.tensor(labels_all, device=device)

        preds_list  = [torch.zeros_like(preds_tensor)  for _ in range(world_size)]
        labels_list = [torch.zeros_like(labels_tensor) for _ in range(world_size)]

        dist.all_gather(preds_list,  preds_tensor)
        dist.all_gather(labels_list, labels_tensor)

        preds_all  = torch.cat(preds_list).cpu().tolist()
        labels_all = torch.cat(labels_list).cpu().tolist()

        # Average loss across ranks (reduce to rank 0 then broadcast, or all_reduce)
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor.item() / world_size

    accuracy = accuracy_score(labels_all, preds_all)
    f1       = f1_score(labels_all, preds_all, average='macro')
    report   = classification_report(
        labels_all, preds_all,
        target_names=LABEL_NAMES, digits=4, zero_division=0,
    )
    return {
        "loss":     total_loss / len(dataloader),
        "accuracy": accuracy,
        "f1_macro": f1,
    }, report, preds_all, labels_all


# ── Training one epoch ─────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device,
                    supcon_fn, supcon_weight, acc_steps):
    model.train()
    tracker = {"loss": 0.0, "task_loss": 0.0, "supcon_loss": 0.0}
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        feats          = batch["extra_features"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            _, task_loss, combined = model(input_ids, attention_mask, feats, labels=labels)
            if task_loss.dim() > 0:
                task_loss = task_loss.mean()

            supcon_loss = torch.tensor(0.0, device=device)
            if supcon_fn is not None:
                emb = nn.functional.normalize(combined, dim=1)
                supcon_loss = supcon_fn(emb, labels)

            total_loss = (task_loss + supcon_weight * supcon_loss) / acc_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % acc_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        tracker["loss"]       += total_loss.item() * acc_steps
        tracker["task_loss"] += task_loss.item()
        tracker["supcon_loss"] += supcon_loss.item()
        pbar.set_postfix({
            "Loss": f"{total_loss.item() * acc_steps:.3f}",
            "SupCon": f"{supcon_loss.item():.3f}" if supcon_fn else "off",
        })

    n = len(dataloader)
    return {k: v / n for k, v in tracker.items()}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Build config (CLI overrides)
    config = DEFAULT_CONFIG.copy()
    for section in DEFAULT_CONFIG:
        config[section] = {**DEFAULT_CONFIG[section]}
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    train_cfg    = config["training"]
    data_cfg     = config["data"]
    seed         = config["common"]["seed"]
    set_seed(seed)

    # Device
    if _is_distributed:
        device = torch.device("cuda", _local_rank)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Fix #3: rank-0 only prints ─────────────────────────────────────────
    if _is_rank0:
        print(f"\n{'='*60}")
        print(f"  SemEval Task A — DDP Training")
        print(f"{'='*60}")
        print(f"  Distributed: Yes  |  World Size: {dist.get_world_size()}"
              if _is_distributed else "\n  Distributed: No")
        print(f"  Device: {device}  |  Config: epochs={train_cfg['num_epochs']}  "
              f"batch_size={train_cfg['batch_size']}  lr={train_cfg['learning_rate']}  "
              f"supcon={train_cfg['use_supcon']}")
        print(f"{'='*60}\n")

    # ── Tokenizer (UnixCoder — from local model/) ─────────────────────────────
    base_model_path = config["model"]["base_model"]
    tokenizer  = __import__("transformers").AutoTokenizer.from_pretrained(base_model_path)

    # ── Data ───────────────────────────────────────────────────────────────────
    data_dir = data_cfg["data_dir"]
    if _is_rank0:
        print(f"Loading data from: {data_dir}")
    train_df = pd.read_parquet(os.path.join(data_dir, "train_processed_cleaned.parquet"))
    val_df   = pd.read_parquet(os.path.join(data_dir, "val_processed_cleaned.parquet"))

    train_df = train_df.dropna(subset=["label"]).reset_index(drop=True)
    val_df   = val_df.dropna(subset=["label"]).reset_index(drop=True)

    if _is_rank0:
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}\n")

    train_ds = AgnosticDataset(train_df, tokenizer, max_length=data_cfg["max_length"], is_train=True)
    val_ds   = AgnosticDataset(val_df,   tokenizer, max_length=data_cfg["max_length"], is_train=False)

    train_dl = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=False,  num_workers=2, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(train_ds, shuffle=True) if _is_distributed else None,
    )
    val_dl = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"] * 2,
        shuffle=False, num_workers=2, pin_memory=True,
        sampler=DistributedSampler(val_ds, shuffle=False) if _is_distributed else None,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = HybridClassifier(config).to(device)
    if _is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[_local_rank], find_unused_parameters=True,
        )
        if _is_rank0:
            print(f"  Model: DDP (world_size={dist.get_world_size()})")
    else:
        if _is_rank0:
            print(f"  Model: Single-device ({device})")

    # ── Optimiser / Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = len(train_dl) * train_cfg["num_epochs"] // train_cfg["gradient_accumulation_steps"]
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler      = GradScaler()

    # ── SupCon loss ────────────────────────────────────────────────────────────
    supcon_fn  = None
    supcon_w   = train_cfg.get("supcon_weight", 0.1)
    if train_cfg.get("use_supcon", False):
        try:
            from pytorch_metric_learning import losses as pml_losses
            supcon_fn = pml_losses.SupConLoss(temperature=0.07).to(device)
            if _is_rank0:
                print(f"  SupCon loss enabled (weight={supcon_w}).")
        except ImportError:
            if _is_rank0:
                print("  [WARN] pytorch_metric_learning not installed — SupCon disabled.")

    # ── Resume ─────────────────────────────────────────────────────────────────
    if args.resume:
        load_checkpoint(model, args.resume, device)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_f1   = 0.0
    patience  = train_cfg["early_stop_patience"]
    patience_counter = 0
    checkpoint_dir   = train_cfg["checkpoint_dir"]
    acc_steps  = train_cfg["gradient_accumulation_steps"]
    num_epochs = train_cfg["num_epochs"]

    if _is_rank0:
        print(f"Starting training for {num_epochs} epochs (patience={patience})...\n")

    for epoch in range(num_epochs):
        if _is_distributed:
            train_dl.sampler.set_epoch(epoch)

        if _is_rank0:
            print(f"Epoch {epoch + 1}/{num_epochs}")

        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device,
            supcon_fn, supcon_w, acc_steps,
        )
        if _is_rank0:
            print(f"  Train: {train_metrics}")

        # Fix #2: gather val metrics from all ranks before comparison
        val_metrics, report, _, _ = evaluate(model, val_dl, device)
        if _is_rank0:
            print(f"  Val:   {val_metrics}")
            print(report)

        current_f1 = val_metrics["f1_macro"]

        # ── Fix #1: rank-0 only checkpoint save ──────────────────────────────
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            if _is_rank0:
                print(f"  --> New best F1: {best_f1:.4f}")
                save_checkpoint(
                    model, tokenizer,
                    os.path.join(checkpoint_dir, "best_model"),
                    epoch, val_metrics, config,
                )
        else:
            patience_counter += 1
            if _is_rank0:
                print(f"  --> No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            if _is_rank0:
                print("\nEARLY STOPPING triggered.")
            break

    if _is_rank0:
        print(f"\nTraining done. Best F1: {best_f1:.4f}")

    if _is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
