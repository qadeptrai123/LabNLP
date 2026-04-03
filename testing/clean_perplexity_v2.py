"""
Clean train_processed.parquet — CORRECTED version.
  1. Remove rows where code is empty-string
  2. Re-compute perplexity for NaN rows with a fixed _compute_perplexity
  3. Drop rows where perp == 0.0 (1-token fragments that can't be scored)
  4. Report before/after stats
"""

import pandas as pd, numpy as np, torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model_path = r'D:\LabNLP\Qwen2.5-Coder-1.5B-Instruct'

# ── fixed perplexity function ──────────────────────────────────────────────────
def fixed_perplexity(code: str, model, tokenizer, max_len: int = 512) -> float:
    if not code.strip():
        return 0.0
    try:
        inputs = tokenizer(
            code, return_tensors="pt", truncation=True, max_length=max_len
        ).to(device)
        with torch.no_grad():
            out = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        loss = out.loss.item()
        if math.isnan(loss) or math.isinf(loss):
            return 0.0
        return loss
    except Exception:
        return 0.0


# ── Load model ────────────────────────────────────────────────────────────────
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

print("=" * 70)
print("ORIGINAL DATASET")
print("=" * 70)
print(f"Shape:          {df.shape}")
arrs = np.stack(df['agnostic_features'].values)
print(f"NaN perplexity: {np.isnan(arrs[:, 0]).sum()}")
print(f"Zero perp:      {(arrs[:, 0] == 0.0).sum()}")
print(f"Empty code:     {(df['code'] == '').sum()}")
print(f"All-zero rows:  {(arrs == 0).all(axis=1).sum()}")

# ── STEP 1: Remove empty-string code rows ────────────────────────────────────
n0 = len(df)
df = df[df['code'] != ''].reset_index(drop=True)
arrs = np.stack(df['agnostic_features'].values)   # re-stack with new index
print(f"\n[Step 1] Removed {(n0 - len(df))} empty-code rows  ->  shape: {df.shape}")

# ── STEP 2: Re-compute perplexity for NaN rows ────────────────────────────────
nan_mask = np.isnan(arrs[:, 0])
nan_idx  = np.where(nan_mask)[0]
print(f"[Step 2] Fixing {len(nan_idx)} NaN perplexity rows...")

fix_log = []
for i, idx in enumerate(nan_idx):
    code  = df.at[idx, 'code']
    gen   = df.at[idx, 'generator']
    old   = arrs[idx, 0]
    new   = fixed_perplexity(code, model, tok, max_len)
    arrs[idx, 0] = new                              # write to array
    fix_log.append((i+1, idx, gen, repr(code), old, new))

print()
print("NaN rows — old -> new:")
print(f"  {'#':>3} {'idx':>7} {'generator':40} {'code':12} {'old':>8} {'new':>10}")
print("  " + "-" * 90)
for row in fix_log:
    num, idx, gen, code, old, new = row
    print(f"  {num:2d} {idx:7d} {gen:40s} {code:12s} {'NaN':>8} {new:10.6f}")

nan_after_fix = np.isnan(arrs[:, 0]).sum()
print(f"\n  -> NaN remaining after fix: {nan_after_fix}")

# Write the FIXED arrs back to df BEFORE dropping zero rows
# This is the CRITICAL fix: reassign the column so df reflects the changes
df['agnostic_features'] = arrs.tolist()
print(f"  -> Wrote fixed features back to df['agnostic_features']")

# ── STEP 3: Drop zero-perplexity rows (1-token fragments) ────────────────────
# Re-stack because df['agnostic_features'] is now list-of-lists
arrs_post = np.stack(df['agnostic_features'].values)
zero_mask = (arrs_post[:, 0] == 0.0)
print(f"\n[Step 3] Dropping {zero_mask.sum()} zero-perplexity rows (1-token fragments)...")

if zero_mask.sum() > 0:
    print("  Zero-perp rows:")
    for idx in np.where(zero_mask)[0]:
        print(f"    idx={idx:7d}  code={repr(df.at[idx,'code']):15s}  gen={df.at[idx,'generator']}")

df = df[~zero_mask].reset_index(drop=True)
arrs_final = np.stack(df['agnostic_features'].values)

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("CLEANED DATASET — FINAL REPORT")
print("=" * 70)
print(f"Shape:          {df.shape}")
print(f"NaN perplexity: {np.isnan(arrs_final[:, 0]).sum()}")
print(f"Zero perp:      {(arrs_final[:, 0] == 0.0).sum()}")
print(f"All-zero rows:  {(arrs_final == 0).all(axis=1).sum()}")
print(f"Empty code:     {(df['code'] == '').sum()}")

print()
print("Perplexity distribution:")
print(f"  Count: {len(arrs_final[:, 0])}")
print(f"  Mean:  {arrs_final[:, 0].mean():.6f}")
print(f"  Std:   {arrs_final[:, 0].std():.6f}")
print(f"  Min:   {arrs_final[:, 0].min():.6f}")
print(f"  P50:   {np.percentile(arrs_final[:, 0], 50):.6f}")
print(f"  P95:   {np.percentile(arrs_final[:, 0], 95):.6f}")
print(f"  Max:   {arrs_final[:, 0].max():.6f}")

print()
print("All 11 features — final stats:")
feat_names = ['perplexity','id_len_avg','id_entropy','id_short_ratio','id_num_ratio',
              'style_consistency','spacing_ratio','line_len_std','ttr','comment_ratio','human_markers']
print(f"  {'Feature':22s} {'Count':>8} {'Min':>10} {'Max':>12} {'Mean':>10} {'Std':>8} {'NaN':>4}")
print("  " + "-" * 76)
for i, n in enumerate(feat_names):
    col = arrs_final[:, i]
    print(f"  {n:20s} {len(col):8d} {col.min():10.4f} {col.max():12.4f} {col.mean():10.4f} {col.std():8.4f} {int(np.isnan(col).sum()):4d}")

print()
print("Label distribution:")
print(f"  Human (label=0): {(df['label']==0).sum():>7d}  ({100*(df['label']==0).mean():.1f}%)")
print(f"  AI     (label=1): {(df['label']==1).sum():>7d}  ({100*(df['label']==1).mean():.1f}%)")

print()
print("Generator distribution:")
print(df['generator'].value_counts().to_string())

print()
print("Language distribution:")
print(df['language'].value_counts().to_string())

# ── SAVE ─────────────────────────────────────────────────────────────────────
out_path = r"D:\LabNLP\SemEval-2026-Task-13\data\Task_A_Processed\train_processed_cleaned.parquet"
df.to_parquet(out_path)
print(f"\nSaved: {out_path}")

# ── VERIFY saved file ────────────────────────────────────────────────────────
df_verify = pd.read_parquet(out_path)
arrs_verify = np.stack(df_verify['agnostic_features'].values)
print()
print("Verification of saved file:")
print(f"  Shape:    {df_verify.shape}")
print(f"  NaN:      {np.isnan(arrs_verify[:, 0]).sum()}")
print(f"  Zero:     {(arrs_verify[:, 0] == 0.0).sum()}")
print(f"  All-zero: {(arrs_verify == 0).all(axis=1).sum()}")
assert np.isnan(arrs_verify[:, 0]).sum() == 0, "SAVING FAILED — NaN still present!"
assert (arrs_verify[:, 0] == 0.0).sum() == 0,  "SAVING FAILED — zero perp still present!"
assert (arrs_verify == 0).all(axis=1).sum() == 0, "SAVING FAILED — all-zero rows still present!"
print("  ✅ All assertions passed — file is clean!")
