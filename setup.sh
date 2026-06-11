#!/bin/bash
# setup.sh — Install polsess_separation on a fresh cloud GPU instance (Vast.ai / RunPod)
#
# Provisions the project (repo + Python deps + env vars). The dataset is handled
# separately by download_dataset.sh — run that after this script.
#
# Prerequisites:
#   - Start from a PyTorch template image (e.g. pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # repo + deps + env vars
#   ./setup.sh --rclone     # also install rclone and configure Google Drive remote

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

# ============================================================================
# Parse arguments
# ============================================================================

SETUP_RCLONE=false
for arg in "$@"; do
    case $arg in
        --rclone)   SETUP_RCLONE=true ;;
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
# rclone (Google Drive backup, opt-in via --rclone)
# ============================================================================

if [ "$SETUP_RCLONE" = true ]; then
    info "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
    ok "rclone installed ($(rclone --version | head -1))"

    if rclone listremotes 2>/dev/null | grep -q "^gdrive:"; then
        ok "rclone gdrive remote already configured — skipping"
    else
        echo ""
        echo "  ─────────────────────────────────────────────────────────────────"
        echo "  rclone needs to authenticate with Google Drive."
        echo "  Because this instance has no browser, you must authorize on your"
        echo "  local PC:"
        echo ""
        echo "    1. Install rclone locally (rclone.org/install) if not already"
        echo "    2. Run on your LOCAL machine:"
        echo "         rclone authorize \"drive\""
        echo "    3. A browser window will open — log in and allow access"
        echo "    4. Copy the token JSON that appears in your local terminal"
        echo "    5. Paste it when rclone config asks for it below"
        echo "  ─────────────────────────────────────────────────────────────────"
        echo ""
        info "Starting rclone config — when prompted:"
        echo "   n  (new remote)"
        echo "   gdrive  (name)"
        echo "   drive   (storage type, or pick from list)"
        echo "   <Enter> for client_id, client_secret, scope 1, root_folder, service_account"
        echo "   n  (don't edit advanced config)"
        echo "   n  (no auto-config — this is a headless machine)"
        echo "   paste the token from your local machine"
        echo "   y  (confirm)"
        echo "   q  (quit config)"
        echo ""
        rclone config

        if rclone listremotes 2>/dev/null | grep -q "^gdrive:"; then
            ok "gdrive remote configured"
        else
            warn "gdrive remote not found after config — run 'rclone config' manually to retry"
        fi
    fi

    echo ""
    echo "  To back up checkpoints and wandb logs to Google Drive:"
    echo "    rclone copy $PROJECT_DIR/checkpoints gdrive:polsess_backups/\$(hostname)/checkpoints --progress"
    echo "    rclone copy $PROJECT_DIR/wandb        gdrive:polsess_backups/\$(hostname)/wandb        --progress"
    echo ""
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
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

if [ -d "$POLSESS_DATA_ROOT" ]; then
    ok "PolSESS dataset found at $POLSESS_DATA_ROOT"
else
    warn "PolSESS dataset not found at $POLSESS_DATA_ROOT — run ./download_dataset.sh"
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo "============================================"
ok "Setup complete!"
echo "============================================"
echo ""
echo "Next:"
echo "  ./download_dataset.sh           # fetch PolSESS into $DATASETS_DIR"
echo "  source ~/.bashrc                # pick up POLSESS_DATA_ROOT"
echo ""
echo "Quick start:"
echo "  cd $PROJECT_DIR"
echo "  python train.py --config experiments/spmamba/spmamba_baseline.yaml"
echo ""
echo "Run tests:"
echo "  pytest"
echo ""
