#!/bin/bash
# ==============================================================================
# SemEval 2026 Task-A — GPU VPS Full Setup Script  (Ubuntu 20.04 / 22.04 / 24.04)
# ==============================================================================
# What it installs & sets up:
#   • System: git, git-lfs, build-essential, curl, wget
#   • Miniconda (latest, Python 3.10)
#   • CUDA 12.1 via conda (pytorch, torchvision, torchaudio)
#   • All Python packages from requirements.txt + extras
#   • Clones this repo, runs download_models.py, creates project dirs
#
# Usage (paste into terminal on your GPU VPS):
#   bash <(curl -fsSL https://your-gist-url-or-local-path/install_gpu_vps.sh)
#
# Or copy this file to your VPS and run:
#   chmod +x install_gpu_vps.sh && ./install_gpu_vps.sh
# ==============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m"

log()   { echo -e "${GREEN}[✔]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✘]${NC} $1"; }
info()  { echo -e "${CYAN}[i]${NC} $1"; }
step()  { echo -e "\n${BOLD}${CYAN}━━━ $1 ━━━${NC}"; }

# ── Config ────────────────────────────────────────────────────────────────────
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
CONDA_ROOT="${HOME}/miniconda3"

# EDIT THIS: URL of your git repo (or leave empty to skip clone)
REPO_URL="${REPO_URL:-}"
# EDIT THIS: branch to checkout (default: main)
REPO_BRANCH="${REPO_BRANCH:-main}"
# EDIT THIS: absolute path where you want the repo cloned
# (defaults to $HOME/semeval-2026-task13)
REPO_DEST="${REPO_DEST:-${HOME}/semeval-2026-task13}"

ENV_NAME="semeval"
PYTHON_VERSION="3.10"

# Kaggle API — set your credentials here OR leave empty to configure later
KAGGLE_USERNAME="${KAGGLE_USERNAME:-}"
KAGGLE_KEY="${KAGGLE_KEY:-}"

# Weights & Biases — set your API key here OR leave empty to skip
WANDB_API_KEY="${WANDB_API_KEY:-}"

# ── Detect GPU ─────────────────────────────────────────────────────────────────
detect_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        log "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total \
                   --format=csv,noheader 2>/dev/null || true
        return 0
    else
        warn "No NVIDIA GPU found via nvidia-smi."
        warn "Script will continue — PyTorch CPU fallback will be used for training."
        return 1
    fi
}

# ── Detect OS ─────────────────────────────────────────────────────────────────
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_VERSION="${VERSION_ID:-}"
    else
        OS_ID="unknown"
        OS_VERSION=""
    fi
    log "OS: ${OS_ID} ${OS_VERSION}"
}

# ── System packages ──────────────────────────────────────────────────────────
install_system_deps() {
    step "1/10 — System dependencies"

    info "Updating apt cache..."
    sudo apt-get update -qq

    info "Installing git, git-lfs, build-essential, curl, wget, unzip..."
    sudo apt-get install -y -qq \
        git git-lfs build-essential curl wget unzip \
        libgl1-mesa-glx libglib2.0-0  # needed by some pip packages (Pillow, etc.)

    # Configure git-lfs
    git lfs install --quiet 2>/dev/null || true
    log "Git LFS configured."
}

# ── Miniconda ─────────────────────────────────────────────────────────────────
install_miniconda() {
    step "2/10 — Miniconda"

    if [ -d "${CONDA_ROOT}" ]; then
        log "Miniconda already installed at ${CONDA_ROOT}"
    else
        info "Downloading Miniconda (latest)..."
        wget -q -O "${MINICONDA_INSTALLER}" "${MINICONDA_URL}"
        info "Installing Miniconda to ${CONDA_ROOT}..."
        bash "${MINICONDA_INSTALLER}" -b -p "${CONDA_ROOT}"
        rm -f "${MINICONDA_INSTALLER}"
        log "Miniconda installed."
    fi

    # Source conda
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    __conda_setup="$("${CONDA_ROOT}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
    eval "${__conda_setup}"

    # Auto-activate base env (quiet)
    conda config --set auto_activate_base false --quiet 2>/dev/null || true

    log "Conda ready: $(conda --version)"
}

# ── Conda environment ────────────────────────────────────────────────────────
setup_conda_env() {
    step "3/10 — Conda environment (${ENV_NAME})"

    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate

    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "Environment '${ENV_NAME}' exists. Recreating it for a clean slate..."
        conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
    fi

    info "Creating environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" \
        python=${PYTHON_VERSION} \
        pytorch \
        torchvision \
        torchaudio \
        pytorch-cuda=12.1 \
        numpy \
        pandas \
        scikit-learn \
        matplotlib \
        seaborn \
        tqdm \
        pyyaml \
        scipy \
        pip \
        -c pytorch -c nvidia -c conda-forge \
        -y

    log "Conda environment '${ENV_NAME}' created."

    # Verify PyTorch + CUDA
    conda activate "${ENV_NAME}"
    python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}'); \
                [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

    # Upgrade pip
    pip install --upgrade pip wheel setuptools
}

# ── pip packages ──────────────────────────────────────────────────────────────
install_pip_packages() {
    step "4/10 — Python packages (pip)"

    conda activate "${ENV_NAME}"

    info "Installing core packages..."
    pip install \
        transformers>=4.48.0 \
        accelerate>=1.0.0 \
        datasets \
        peft \
        bitsandbytes \
        pytorch-metric-learning \
        safetensors \
        tokenizers \
        sentencepiece \
        rich \
        regex \
        protobuf \
        scipy

    info "Installing utility packages..."
    pip install \
        python-dotenv \
        comet_ml \
        kaggle \
        wandb \
        ipython

    info "Installing GPU extras for bitsandbytes (reinstall with correct CUDA)..."
    # bitsandbytes pre-built wheels are CUDA-version-sensitive;
    # if import fails, the line below forces a source build (slow).
    python -c "import bitsandbytes; print(f'  bitsandbytes {bitsandbytes.__version__} OK')" 2>/dev/null || \
        pip install bitsandbytes --force-reinstall --no-cache-dir 2>/dev/null || true

    log "All pip packages installed."
    pip list | grep -E "transformers|torch|accelerate|bitsandbytes|peft"
}

# ── Verify GPU ────────────────────────────────────────────────────────────────
verify_gpu() {
    step "5/10 — GPU verification"

    conda activate "${ENV_NAME}"

    python - <<'PYEOF'
import torch
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version   : {torch.version.cuda}")
    print(f"  GPU count      : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | "
              f"{round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)} GB")
    # Quick matmul test
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    _ = (x @ y).cpu()
    print("  GEMM test PASSED")
else:
    print("  WARNING: No CUDA — training will run on CPU (very slow)")
PYEOF
    log "GPU verification done."
}

# ── Clone / sync repo ────────────────────────────────────────────────────────
sync_repo() {
    step "6/10 — Repository"

    if [ -z "${REPO_URL}" ]; then
        warn "REPO_URL is not set — skipping git clone."
        warn "Set the env var or manually git-clone your repo into ${REPO_DEST}"
        info "Example:  git clone https://github.com/you/semeval-2026-task13.git ${REPO_DEST}"
        return 0
    fi

    if [ -d "${REPO_DEST}/.git" ]; then
        info "Repo already exists at ${REPO_DEST} — pulling latest..."
        cd "${REPO_DEST}"
        git -C "${REPO_DEST}" pull origin "${REPO_BRANCH}"
    else
        info "Cloning ${REPO_URL} into ${REPO_DEST} ..."
        git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${REPO_DEST}"
    fi

    log "Repo ready at ${REPO_DEST}"
}

# ── Git LFS ──────────────────────────────────────────────────────────────────
setup_git_lfs() {
    step "7/10 — Git LFS"

    conda activate "${ENV_NAME}"

    # Track typical model file extensions
    info "Configuring Git LFS to track model files..."
    git lfs install 2>/dev/null || true
    git lfs track "*.bin" 2>/dev/null || true
    git lfs track "*.pt" 2>/dev/null || true
    git lfs track "*.pth" 2>/dev/null || true
    git lfs track "*.safetensors" 2>/dev/null || true
    git lfs track "*.gguf" 2>/dev/null || true
    git lfs track "*.h5" 2>/dev/null || true
    git lfs track "*.onnx" 2>/dev/null || true
    git lfs track "*.parquet" 2>/dev/null || true
    git lfs track "*.csv" 2>/dev/null || true
    git lfs track "*.gz" 2>/dev/null || true
    log "Git LFS tracking configured."
}

# ── Project structure ─────────────────────────────────────────────────────────
create_project_structure() {
    step "8/10 — Project structure"

    if [ ! -d "${REPO_DEST}" ]; then
        warn "REPO_DEST '${REPO_DEST}' not found — creating it as a placeholder..."
        mkdir -p "${REPO_DEST}"
    fi

    info "Creating directory structure..."
    mkdir -p "${REPO_DEST}/data/Task_A"
    mkdir -p "${REPO_DEST}/data/Task_A_Processed"
    mkdir -p "${REPO_DEST}/model"
    mkdir -p "${REPO_DEST}/checkpoints"
    mkdir -p "${REPO_DEST}/src"
    mkdir -p "${REPO_DEST}/results"

    # Copy .env template if not present
    if [ ! -f "${REPO_DEST}/.env" ]; then
        cat > "${REPO_DEST}/.env" <<'ENVEOF'
# ── PATHS ────────────────────────────────────────────────────────────────────
DATA_PATH=./data
IMG_PATH=./img

# ── COMET ML (optional) ───────────────────────────────────────────────────────
COMET_API_KEY=your_comet_api_key_here
COMET_PROJECT_NAME=semeval-2026-task13
COMET_WORKSPACE=your_workspace_here

# ── WANDB (optional) ─────────────────────────────────────────────────────────
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=semeval-2026-task13
WANDB_ENTITY=your_entity_here

# ── KAGGLE (optional) ────────────────────────────────────────────────────────
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
ENVEOF
        log "Created .env template at ${REPO_DEST}/.env — please edit it."
    fi

    # Copy kaggle.json if provided via env
    if [ -n "${KAGGLE_USERNAME}" ] && [ -n "${KAGGLE_KEY}" ]; then
        mkdir -p "${HOME}/.kaggle"
        echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" \
            > "${HOME}/.kaggle/kaggle.json"
        chmod 600 "${HOME}/.kaggle/kaggle.json"
        log "Kaggle credentials configured."
    fi

    # Set WANDB API key if provided
    if [ -n "${WANDB_API_KEY}" ]; then
        export WANDB_API_KEY
        conda activate "${ENV_NAME}"
        pip install -q wandb
        python -c "import wandb; wandb.login(key='${WANDB_API_KEY}')" 2>/dev/null || true
        log "W&B API key configured."
    fi

    log "Project structure ready."
    ls -la "${REPO_DEST}/"
}

# ── Download HuggingFace models ──────────────────────────────────────────────
download_models() {
    step "9/10 — Downloading HuggingFace models"

    conda activate "${ENV_NAME}"

    if [ ! -f "${REPO_DEST}/download_models.py" ]; then
        warn "download_models.py not found in ${REPO_DEST} — skipping auto-download."
        warn "Run manually after placing the script:"
        warn "  conda activate ${ENV_NAME}"
        warn "  python download_models.py"
        return 0
    fi

    info "Models will be saved to ${REPO_DEST}/model/"
    info "This may take 10-30 minutes depending on your connection speed."
    info "Two models will be downloaded:"
    info "  1. microsoft/unixcoder-base      (~400 MB)"
    info "  2. Qwen/Qwen2.5-Coder-1.5B-Instruct  (~3 GB)"

    # Run with conda run to ensure env is active
    conda run -n "${ENV_NAME}" python "${REPO_DEST}/download_models.py"

    log "Model download step complete."
}

# ── Final verification ────────────────────────────────────────────────────────
final_check() {
    step "10/10 — Final verification"

    conda activate "${ENV_NAME}"

    echo ""
    info "Verifying all key imports..."
    python - <<'PYEOF'
try:
    import torch;           print(f"  torch           {torch.__version__}")
except ImportError as e:     print(f"  torch           FAIL: {e}")
try:
    import transformers;    print(f"  transformers    {transformers.__version__}")
except ImportError as e:     print(f"  transformers    FAIL: {e}")
try:
    import accelerate;     print(f"  accelerate      {accelerate.__version__}")
except ImportError as e:    print(f"  accelerate      FAIL: {e}")
try:
    import peft;            print(f"  peft            {peft.__version__}")
except ImportError as e:    print(f"  peft            FAIL: {e}")
try:
    import bitsandbytes;    print(f"  bitsandbytes    {bitsandbytes.__version__}")
except ImportError as e:    print(f"  bitsandbytes    FAIL: {e}")
try:
    import pytorch_metric_learning; print(f"  pytorch_metric_learning OK")
except ImportError as e:    print(f"  pytorch_metric_learning FAIL: {e}")
try:
    import pandas;          print(f"  pandas          {pandas.__version__}")
except ImportError as e:    print(f"  pandas          FAIL: {e}")
try:
    import sklearn;         print(f"  scikit-learn    {sklearn.__version__}")
except ImportError as e:    print(f"  scikit-learn    FAIL: {e}")
try:
    import yaml;            print(f"  pyyaml          OK")
except ImportError as e:    print(f"  pyyaml          FAIL: {e}")
try:
    import comet_ml;        print(f"  comet_ml        OK")
except ImportError as e:    print(f"  comet_ml        FAIL: {e}")
try:
    import kaggle;          print(f"  kaggle          OK")
except ImportError as e:    print(f"  kaggle          FAIL: {e}")
PYEOF

    echo ""
    if [ -d "${REPO_DEST}/model/unixcoder-base" ]; then
        log "UnixCoder model: found at ${REPO_DEST}/model/unixcoder-base"
    else
        warn "UnixCoder model: NOT found — run 'python download_models.py' later"
    fi

    if [ -d "${REPO_DEST}/model/Qwen2.5-Coder-1.5B-Instruct" ]; then
        log "Qwen model: found at ${REPO_DEST}/model/Qwen2.5-Coder-1.5B-Instruct"
    else
        warn "Qwen model: NOT found — run 'python download_models.py' later"
    fi
}

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  SemEval 2026 Task-A — GPU VPS Setup${NC}"
echo -e "${BOLD}${CYAN}  $(date '+%Y-%m-%d')${NC}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${NC}"
echo ""

# ── Pre-flight ───────────────────────────────────────────────────────────────
detect_os
detect_gpu

echo ""
info "Repo will be cloned to : ${REPO_DEST}"
info "Conda env name         : ${ENV_NAME}"
info "Python version         : ${PYTHON_VERSION}"
echo ""

# ── Run all steps ────────────────────────────────────────────────────────────
install_system_deps
install_miniconda
setup_conda_env
install_pip_packages
verify_gpu
sync_repo
setup_git_lfs
create_project_structure
download_models
final_check

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  SETUP COMPLETE${NC}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Activate the environment:"
echo -e "    ${BOLD}conda activate ${ENV_NAME}${NC}"
echo ""
echo -e "  Navigate to the repo:"
echo -e "    ${BOLD}cd ${REPO_DEST}${NC}"
echo ""
echo -e "  Download models (if not already done):"
echo -e "    ${BOLD}python download_models.py${NC}"
echo ""
echo -e "  Start training (single GPU):"
echo -e "    ${BOLD}python train.py${NC}"
echo ""
echo -e "  Start training (multi-GPU, e.g. 2 GPUs):"
echo -e "    ${BOLD}torchrun --nproc_per_node=2 train.py${NC}"
echo ""
echo -e "  Optional — edit .env for API keys:"
echo -e "    ${BOLD}nano ${REPO_DEST}/.env${NC}"
echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
