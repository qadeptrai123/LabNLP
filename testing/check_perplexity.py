import pandas as pd, numpy as np, torch, math, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model_path = r'D:\LabNLP\Qwen2.5-Coder-1.5B-Instruct'

print('Loading Qwen2.5-Coder-1.5B-Instruct...')
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
).eval()
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
max_len = 512
print('Model ready.\n')

# ── Load dataset ──────────────────────────────────────────────────────────────
path = r'D:\LabNLP\SemEval-2026-Task-13\data\Task_A_Processed\train_processed.parquet'
df = pd.read_parquet(path)
arrs = np.stack(df['agnostic_features'].values)

def recompute_loss(code):
    if not code.strip():
        return None, 0, 'empty_guard'
    n_tok = len(tok.encode(code))
    try:
        inputs = tok(code, return_tensors='pt', truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        loss = out.loss.item()
        return loss, n_tok, 'ok'
    except Exception as e:
        return None, n_tok, f'error:{e}'

# ── TEST A: Normal rows at diverse percentiles ────────────────────────────────
print('=' * 70)
print('TEST A — Normal rows: stored vs re-computed loss')
print('=' * 70)
normal_idx = np.where(~np.isnan(arrs[:, 0]))[0]
for pct in [5, 25, 50, 75, 95]:
    row_idx = int(np.percentile(normal_idx, pct))
    code    = df.at[row_idx, 'code']
    stored  = arrs[row_idx, 0]
    loss, n_tok, status = recompute_loss(code)
    if status == 'ok':
        diff  = abs(stored - loss)
        label = 'OK' if diff < 0.01 else 'MISMATCH'
        print(f'  row {row_idx:6d}  stored={stored:.6f}  computed={loss:.6f}  diff={diff:.8f}  [{label}]')
        print(f'           code[:60]: {repr(code[:60])}')

# ── TEST B: All NaN rows ─────────────────────────────────────────────────────
print()
print('=' * 70)
print('TEST B — NaN rows: why do they become NaN?')
print('=' * 70)
nan_perp_mask = np.isnan(arrs[:, 0])
nan_rows = df[nan_perp_mask].reset_index()
nan_arrs  = arrs[nan_perp_mask]

for i in range(len(nan_rows)):
    code   = nan_rows.at[i, 'code']
    gen    = nan_rows.at[i, 'generator']
    stored = nan_arrs[i, 0]
    loss, n_tok, status = recompute_loss(code)
    if status == 'ok':
        print(f'  [{i+1:02d}] gen={gen[:35]:35s}  code={repr(code):12s}  n_tok={n_tok:3d}  stored=NaN  computed={loss:.8f}  isNaN={math.isnan(loss)}')
    else:
        print(f'  [{i+1:02d}] gen={gen[:35]:35s}  code={repr(code):12s}  n_tok={n_tok:3d}  stored=NaN  status={status}')

# ── TEST C: Zero-perplexity row ──────────────────────────────────────────────
print()
print('=' * 70)
print('TEST C — Zero-perplexity row')
print('=' * 70)
zero_perp_idx = np.where(arrs[:, 0] == 0.0)[0]
for idx in zero_perp_idx:
    code   = df.at[idx, 'code']
    stored = arrs[idx, 0]
    loss, n_tok, status = recompute_loss(code)
    if status == 'ok':
        print(f'  row {idx:6d}  n_tok={n_tok}  stored={stored:.6f}  computed={loss:.8f}')
    else:
        print(f'  row {idx:6d}  stored={stored:.6f}  status={status}')
    print(f'  code: {repr(code)}')
    print(f'  features: {arrs[idx]}')

# ── TEST D: All-zero feature vector row ──────────────────────────────────────
print()
print('=' * 70)
print('TEST D — All-zero feature vector row')
print('=' * 70)
zero_vec_mask = (arrs == 0).all(axis=1)
zero_vec_idx  = np.where(zero_vec_mask)[0]
for idx in zero_vec_idx:
    code   = df.at[idx, 'code']
    stored = arrs[idx, 0]
    loss, n_tok, status = recompute_loss(code)
    if status == 'ok':
        print(f'  row {idx:6d}  n_tok={n_tok}  stored={stored:.6f}  computed={loss:.8f}')
    else:
        print(f'  row {idx:6d}  stored={stored:.6f}  status={status}')
    print(f'  code: {repr(code)}')
    print(f'  features: {arrs[idx]}')

# ── TEST E: Edge-case short strings ─────────────────────────────────────────
print()
print('=' * 70)
print('TEST E — Short strings: at what token length does NaN appear?')
print('=' * 70)
test_strings = [
    ('#',        'single char `#`'),
    ('{',        'single char `{`'),
    ('A',        'single char `A`'),
    ('4',        'digit `4`'),
    ('no',       '2-char `no`'),
    ('NO',       '2-char `NO`'),
    ('True',     '4-char `True`'),
    ('status',   '6-char `status`'),
    ('columns',  '7-char `columns`'),
    ('def foo',  '8-char `def foo`'),
    ('def foo():',  '12-char `def foo():`'),
    ('def foo():\n    return 0\n', 'multi-line'),
]
for code, label in test_strings:
    n_tok = len(tok.encode(code))
    inputs = tok(code, return_tensors='pt', truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
    loss = out.loss.item()
    flag = ' *** NaN ***' if math.isnan(loss) else ''
    print(f'  {label:30s}  n_tok={n_tok:3d}  loss={loss:.8f}{flag}')

# ── TEST F: Cross-check loss vs perplexity semantics ─────────────────────────
print()
print('=' * 70)
print('TEST F — Verify: is stored value loss or perplexity (exp(loss))?')
print('=' * 70)
# Take a few normal rows, compute both
for row_idx in [0, 100, 1000, 10000]:
    code    = df.at[row_idx, 'code']
    stored  = arrs[row_idx, 0]
    inputs  = tok(code, return_tensors='pt', truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
    loss     = out.loss.item()
    perplexity = math.exp(loss)
    diff_loss     = abs(stored - loss)
    diff_perplexity = abs(stored - perplexity)
    print(f'  row {row_idx:6d}  stored={stored:.6f}')
    print(f'           diff_vs_loss={diff_loss:.8f}  diff_vs_perplexity={diff_perplexity:.8f}')
    print(f'           -> closest: {"loss (cross-entropy)" if diff_loss < diff_perplexity else "perplexity = exp(loss)"}')

print()
print('Done.')
