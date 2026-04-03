"""Clean val_processed.parquet — same logic as train_processed_cleaned."""
import pandas as pd, numpy as np, torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model_path = r'D:\LabNLP\Qwen2.5-Coder-1.5B-Instruct'

def fixed_perplexity(code: str, model, tokenizer, max_len: int = 512) -> float:
    if not code.strip():
        return 0.0
    try:
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        loss = out.loss.item()
        return 0.0 if (math.isnan(loss) or math.isinf(loss)) else loss
    except Exception:
        return 0.0

print("Loading Qwen2.5-Coder-1.5B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
).eval()
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("Ready.\n")

# ── Load val ─────────────────────────────────────────────────────────────────
path = r"D:\LabNLP\SemEval-2026-Task-13\data\Task_A_Processed\val_processed.parquet"
df = pd.read_parquet(path)
arrs = np.stack(df['agnostic_features'].values).copy()

print("=== ORIGINAL VAL ===")
print(f"Shape:          {df.shape}")
print(f"NaN perplexity: {np.isnan(arrs[:, 0]).sum()}")
print(f"Zero perp:      {(arrs[:, 0] == 0.0).sum()}")
print(f"Empty code:     {(df['code'] == '').sum()}")

# Step 1: empty code (none expected)
n0 = len(df)
df = df[df['code'] != ''].reset_index(drop=True)
if n0 - len(df):
    arrs = np.stack(df['agnostic_features'].values).copy()
    print(f"[Step 1] Removed {n0 - len(df)} empty-code rows -> shape: {df.shape}")
else:
    print("[Step 1] No empty-code rows found.")

# Step 2: fix NaN perplexity
nan_mask = np.isnan(arrs[:, 0])
nan_idx  = np.where(nan_mask)[0]
print(f"[Step 2] Fixing {len(nan_idx)} NaN perplexity rows...")

for idx in nan_idx:
    code = df.at[idx, 'code']
    gen  = df.at[idx, 'generator']
    new  = fixed_perplexity(code, model, tok)
    arrs[idx, 0] = new
    print(f"  code={repr(code):15s}  gen={gen:45s}  old=NaN  new={new:.6f}")

# Write back
df['agnostic_features'] = arrs.tolist()
arrs = np.stack(df['agnostic_features'].values)
print(f"  -> NaN remaining: {np.isnan(arrs[:, 0]).sum()}")

# Step 3: drop zero-perp rows
zero_mask = (arrs[:, 0] == 0.0)
if zero_mask.sum():
    print(f"[Step 3] Dropping {zero_mask.sum()} zero-perp rows...")
    df = df[~zero_mask].reset_index(drop=True)
    arrs = np.stack(df['agnostic_features'].values)
else:
    print("[Step 3] No zero-perp rows to drop.")

# ── Final report ──────────────────────────────────────────────────────────────
print()
print("=== CLEANED VAL ===")
print(f"Shape:          {df.shape}")
print(f"NaN perplexity: {np.isnan(arrs[:, 0]).sum()}")
print(f"Zero perp:      {(arrs[:, 0] == 0.0).sum()}")
print(f"All-zero rows:  {(arrs == 0).all(axis=1).sum()}")
print()
print("Perplexity distribution:")
p = arrs[:, 0]
print(f"  Count: {len(p)}")
print(f"  Mean:  {p.mean():.6f}")
print(f"  Std:   {p.std():.6f}")
print(f"  Min:   {p.min():.6f}")
print(f"  P50:   {np.percentile(p, 50):.6f}")
print(f"  P95:   {np.percentile(p, 95):.6f}")
print(f"  Max:   {p.max():.6f}")
print()
print("Label distribution:")
print(f"  Human (label=0): {(df['label']==0).sum():>7d}  ({100*(df['label']==0).mean():.1f}%)")
print(f"  AI     (label=1): {(df['label']==1).sum():>7d}  ({100*(df['label']==1).mean():.1f}%)")
print()
print("Language distribution:")
print(df['language'].value_counts().to_string())

# Save
out_path = r"D:\LabNLP\SemEval-2026-Task-13\data\Task_A_Processed\val_processed_cleaned.parquet"
df.to_parquet(out_path)
print(f"\nSaved: {out_path}")

# Verify
df_v = pd.read_parquet(out_path)
arrs_v = np.stack(df_v['agnostic_features'].values)
print(f"Verification: shape={df_v.shape}  NaN={np.isnan(arrs_v[:,0]).sum()}  zero={(arrs_v[:,0]==0).sum()}  all_zero={(arrs_v==0).all(axis=1).sum()}")
print("PASS" if np.isnan(arrs_v[:,0]).sum()==0 and (arrs_v[:,0]==0).sum()==0 and (arrs_v==0).all(axis=1).sum()==0 else "FAIL")