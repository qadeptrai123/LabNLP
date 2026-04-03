"""
Clean train_processed.parquet:
  1. Remove rows where code is empty-string
  2. Re-compute perplexity for the 34 NaN rows with a fixed _compute_perplexity
     (guard: if loss is NaN/Inf, return 0.0)
  3. Report before/after stats
"""

import pandas as pd, numpy as np, torch, math, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model_path = r'D:\LabNLP\Qwen2.5-Coder-1.5B-Instruct'

# ── fixed perplexity function ──────────────────────────────────────────────────
def fixed_perplexity(code: str, model, tokenizer, max_len: int = 512) -> float:
    """
    Identical to preprocess.py _compute_perplexity BUT with NaN/Inf guard.
    Returns cross-entropy loss, not perplexity.
    """
    if not code.strip():
        return 0.0
    try:
        inputs = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(device)
        with torch.no_grad():
            out = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        loss = out.loss.item()
        # FIX: guard against NaN/Inf loss
        if math.isnan(loss) or math.isinf(loss):
            return 0.0
        return loss
    except Exception:
        return 0.0


# ── Load model once ───────────────────────────────────────────────────────────
print("Loading Qwen2.5-Coder-1.5B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
).eval()
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
max_len = 512
print("Model ready.\n")

# ── Load dataset ──────────────────────────────────────────────────────────────
path = r"D:\LabNLP\SemEval-2026-Task-13\data\Task_A_Processed\train_processed.parquet"
df = pd.read_parquet(path)
arrs = np.stack(df['agnostic_features'].values).copy()

print(f"=== BEFORE CLEANING ===")
print(f"Shape: {df.shape}")
print(f"Total NaN in perplexity: {np.isnan(arrs[:, 0]).sum()}")
print(f"Empty-string code rows: {(df['code'] == '').sum()}")
print(f"All-zero feature rows: {(arrs == 0).all(axis=1).sum()}")
print()

# ── STEP 1: Remove empty-string code rows ─────────────────────────────────────
n_before = len(df)
df = df[df['code'] != ''].reset_index(drop=True)
arrs = np.stack(df['agnostic_features'].values).copy()
n_removed_empty = n_before - len(df)
print(f"Removed {n_removed_empty} empty-string code rows.")
print(f"Shape after removal: {df.shape}\n")

# ── STEP 2: Re-compute perplexity for NaN rows ────────────────────────────────
nan_perp_mask = np.isnan(arrs[:, 0])
nan_indices   = np.where(nan_perp_mask)[0]
print(f"Found {len(nan_indices)} NaN perplexity rows to fix.")

new_values = []
for i, idx in enumerate(nan_indices):
    code  = df.at[idx, 'code']
    gen   = df.at[idx, 'generator']
    old   = arrs[idx, 0]
    new   = fixed_perplexity(code, model, tok, max_len)
    new_values.append((i+1, idx, gen, repr(code), old, new, 'OK' if new != 0.0 or code.strip() == '' else 'ZERO'))
    arrs[idx, 0] = new

print()
print("=== NaN rows re-computed ===")
print(f"{'#':>3} {'idx':>7} {'generator':35} {'code':12} {'old':>8} {'new':>10} {'status'}")
print("-" * 100)
for row in new_values:
    num, idx, gen, code, old, new, status = row
    print(f"  {num:2d} {idx:7d} {gen:35s} {code:12s} {str(old):>8} {new:10.6f}  {status}")

# ── STEP 3: Final stats ────────────────────────────────────────────────────────
nan_remain   = np.isnan(arrs[:, 0]).sum()
zero_remain  = (arrs == 0).all(axis=1).sum()
empty_remain = (df['code'] == '').sum()
inf_remain   = np.isinf(arrs).sum()

print()
print("=== AFTER CLEANING ===")
print(f"Shape:            {df.shape}")
print(f"Total NaN:        {nan_remain}")
print(f"Total Inf:        {inf_remain}")
print(f"Empty-code rows:  {empty_remain}")
print(f"All-zero rows:    {zero_remain}")

print()
print("=== FEATURE STATS AFTER CLEAN ===")
feat_names = ['perplexity','id_len_avg','id_entropy','id_short_ratio','id_num_ratio',
              'style_consistency','spacing_ratio','line_len_std','ttr','comment_ratio','human_markers']
print(f"{'Feature':22s} {'Count':>8} {'Min':>10} {'Max':>12} {'Mean':>10} {'Std':>8} {'NaN':>5}")
print("-" * 80)
for i, name in enumerate(feat_names):
    col = arrs[:, i]
    valid = col[~np.isnan(col)]
    print(f"  {name:20s} {len(valid):8d} {valid.min():10.4f} {valid.max():12.4f} {valid.mean():10.4f} {valid.std():8.4f} {np.isnan(col).sum():5d}")

print()
print("=== PERPLEXITY DISTRIBUTION ===")
valid_perp = arrs[:, 0]
print(f"  Count: {len(valid_perp)}")
print(f"  Mean:   {valid_perp.mean():.6f}")
print(f"  Std:    {valid_perp.std():.6f}")
print(f"  Min:    {valid_perp.min():.6f}")
print(f"  P25:    {np.percentile(valid_perp, 25):.6f}")
print(f"  P50:    {np.percentile(valid_perp, 50):.6f}")
print(f"  P75:    {np.percentile(valid_perp, 75):.6f}")
print(f"  P95:    {np.percentile(valid_perp, 95):.6f}")
print(f"  P99:    {np.percentile(valid_perp, 99):.6f}")
print(f"  Max:    {valid_perp.max():.6f}")

print()
print("=== LABEL DISTRIBUTION ===")
print(df['label'].value_counts())
print(f"  Human (label=0): {(df['label']==0).sum()}")
print(f"  AI     (label=1): {(df['label']==1).sum()}")

print()
print("=== GENERATOR DISTRIBUTION ===")
print(df['generator'].value_counts().to_string())

print()
print("=== LANGUAGE DISTRIBUTION ===")
print(df['language'].value_counts().to_string())

# ── SAVE cleaned parquet ──────────────────────────────────────────────────────
out_path = r"D:\LabNLP\SemEval-2026-Task-13\data\Task_A_Processed\train_processed_cleaned.parquet"
df.to_parquet(out_path)
print()
print(f"Saved cleaned parquet to:")
print(f"  {out_path}")
print(f"  Rows saved: {len(df)}")