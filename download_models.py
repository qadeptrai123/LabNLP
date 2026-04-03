"""
Download models from HuggingFace and save to local model/ directory.

Usage:
    python download_models.py              # download both models
    python download_models.py --qwen       # Qwen only
    python download_models.py --unixcoder # UnixCoder only
    python download_models.py --force     # re-download even if exists
"""
import os
import argparse

# Paths
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(SCRIPT_DIR, "model")
QWEN_DIR      = os.path.join(MODEL_DIR, "Qwen2.5-Coder-1.5B-Instruct")
UNIXCODER_DIR = os.path.join(MODEL_DIR, "unixcoder-base")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Download helpers ──────────────────────────────────────────────────────────
def download_qwen(target_dir: str, force: bool):
    """Download Qwen/Qwen2.5-Coder-1.5B-Instruct from HuggingFace."""
    if os.path.exists(target_dir) and not force:
        print(f"[SKIP] {target_dir} already exists. Use --force to re-download.")
        return

    print(f"Downloading Qwen/Qwen2.5-Coder-1.5B-Instruct ...")
    print(f"  -> {target_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    # Load model + tokenizer from HF, then save locally
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        torch_dtype=dtype,
        device_map=device if device == "cuda" else "cpu",
        trust_remote_code=True,
    )

    print("  Saving locally ...")
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"  Done: {target_dir}")


def download_unixcoder(target_dir: str, force: bool):
    """Download microsoft/unixcoder-base from HuggingFace."""
    if os.path.exists(target_dir) and not force:
        print(f"[SKIP] {target_dir} already exists. Use --force to re-download.")
        return

    print(f"Downloading microsoft/unixcoder-base ...")
    print(f"  -> {target_dir}")

    from transformers import AutoModel, AutoTokenizer

    model     = AutoModel.from_pretrained("microsoft/unixcoder-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")

    print("  Saving locally ...")
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    print(f"  Done: {target_dir}")


def verify(path: str, name: str) -> bool:
    """Check that the local model dir has the expected files."""
    if not os.path.isdir(path):
        return False
    # Minimal required files
    required = ["config.json", "tokenizer_config.json"]
    present  = [f for f in required if os.path.exists(os.path.join(path, f))]
    if len(present) == len(required):
        size_mb = sum(
            os.path.getsize(os.path.join(path, f))
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ) / 1024 ** 2
        print(f"  [OK] {name}: {len(present)}/{len(required)} files, {size_mb:.1f} MB")
        return True
    else:
        print(f"  [INCOMPLETE] {name}: only {len(present)}/{len(required)} files found")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace models to local model/ directory")
    parser.add_argument("--qwen",       action="store_true", help="Download Qwen only")
    parser.add_argument("--unixcoder",  action="store_true", help="Download UnixCoder only")
    parser.add_argument("--force",      action="store_true", help="Re-download even if local copy exists")
    args = parser.parse_args()

    # Default: download both if neither flag set
    do_qwen      = args.qwen      or (not args.qwen and not args.unixcoder)
    do_unixcoder = args.unixcoder or (not args.qwen and not args.unixcoder)

    print(f"Model directory: {MODEL_DIR}")
    print(f"Device: cuda available = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print()

    if do_qwen:
        download_qwen(QWEN_DIR, args.force)
        verify(QWEN_DIR, "Qwen2.5-Coder-1.5B-Instruct")

    if do_unixcoder:
        download_unixcoder(UNIXCODER_DIR, args.force)
        verify(UNIXCODER_DIR, "unixcoder-base")

    print()
    print("=== Summary ===")
    verify(QWEN_DIR,      "Qwen2.5-Coder-1.5B-Instruct")
    verify(UNIXCODER_DIR, "unixcoder-base")
    print()
    print("All done.")


if __name__ == "__main__":
    main()
