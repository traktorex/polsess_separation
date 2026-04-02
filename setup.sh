#!/bin/bash
# setup.sh — Set up polsess_separation on a fresh cloud GPU instance (Vast.ai / RunPod)
#
# Prerequisites:
#   - Start from a PyTorch template image (e.g. pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel)
#   - Upload PolSESS.tar.gz to cloud storage (Google Drive, S3, etc.)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # full setup (repo + deps + dataset)
#   ./setup.sh --no-data    # skip dataset download (if already mounted)

set -euo pipefail

# ============================================================================
# Configuration — edit these before running
# ============================================================================

# Git repository (HTTPS — no SSH keys needed on cloud instances)
REPO_URL="https://github.com/traktorex/polsess_separation.git"
REPO_BRANCH="main"

# Where to install
PROJECT_DIR="$HOME/polsess_separation"
DATASETS_DIR="$HOME/datasets"

# PolSESS dataset archive URL
# Supports: Google Drive (via gdown), direct HTTP/HTTPS, rclone remote paths
# Leave empty to skip download (e.g. if dataset is on a mounted volume).
POLSESS_URL="https://drive.google.com/uc?id=149d8LFyX9jr_AJNwb2BtTcffuuKqUg1B"

# ============================================================================
# Parse arguments
# ============================================================================

SKIP_DATA=false
for arg in "$@"; do
    case $arg in
        --no-data) SKIP_DATA=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ============================================================================
# Helper functions
# ============================================================================

info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERR ]\033[0m $*"; exit 1; }

# ============================================================================
# System checks
# ============================================================================

info "Checking environment..."

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found — is this a GPU instance?"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ok "GPU detected"

python3 --version || error "Python 3 not found"

# ============================================================================
# Clone repository
# ============================================================================

if [ -d "$PROJECT_DIR/.git" ]; then
    info "Repository already cloned, pulling latest..."
    cd "$PROJECT_DIR"
    git pull origin "$REPO_BRANCH"
else
    info "Cloning repository..."
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

ok "Repository ready at $PROJECT_DIR"

# ============================================================================
# Install Python dependencies
# ============================================================================

info "Installing Python dependencies..."
pip install -r requirements.txt

ok "Dependencies installed"

# Verify mamba-ssm (critical for SPMamba)
python3 -c "import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__}')" 2>/dev/null \
    && ok "mamba-ssm available" \
    || warn "mamba-ssm not installed — SPMamba/MambaTasNet/DPMamba will be unavailable"

# ============================================================================
# Set environment variables
# ============================================================================

info "Configuring environment variables..."

export POLSESS_DATA_ROOT="$DATASETS_DIR/PolSESS_C_both/PolSESS_C_both"
export TF_ENABLE_ONEDNN_OPTS=0

# Persist for future shell sessions
grep -q "POLSESS_DATA_ROOT" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc << ENVEOF

# PolSESS separation project
export POLSESS_DATA_ROOT="$DATASETS_DIR/PolSESS_C_both/PolSESS_C_both"
export TF_ENABLE_ONEDNN_OPTS=0
ENVEOF

ok "Environment configured"

# ============================================================================
# Download dataset
# ============================================================================

if [ "$SKIP_DATA" = false ]; then
    if [ -z "$POLSESS_URL" ]; then
        warn "POLSESS_URL is empty — set it at the top of this script or download manually"
        warn "Expected location: $POLSESS_DATA_ROOT"
    else
        mkdir -p "$DATASETS_DIR"
        info "Downloading PolSESS dataset..."

        archive="/tmp/PolSESS.tar.gz"

        if [[ "$POLSESS_URL" == *"drive.google.com"* ]]; then
            pip install -q gdown 2>/dev/null
            gdown "$POLSESS_URL" -O "$archive"
        elif [[ "$POLSESS_URL" == *":"* && "$POLSESS_URL" != "http"* ]]; then
            rclone copy "$POLSESS_URL" /tmp/ --progress
        else
            wget -q --show-progress -O "$archive" "$POLSESS_URL"
        fi

        info "Extracting to $DATASETS_DIR..."
        tar xzf "$archive" -C "$DATASETS_DIR"
        rm -f "$archive"
        ok "PolSESS dataset ready"
    fi
else
    warn "Skipping dataset download (--no-data)"
fi

# ============================================================================
# Verify installation
# ============================================================================

info "Running verification..."

cd "$PROJECT_DIR"

python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

if [ -d "$POLSESS_DATA_ROOT" ]; then
    ok "PolSESS dataset found at $POLSESS_DATA_ROOT"
else
    warn "PolSESS dataset not found at $POLSESS_DATA_ROOT"
    warn "Download it manually or set POLSESS_DATA_ROOT to the correct path"
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo "============================================"
ok "Setup complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo "  cd $PROJECT_DIR"
echo "  python train.py --config experiments/spmamba/spmamba_baseline.yaml"
echo ""
echo "Run tests:"
echo "  pytest"
echo ""
