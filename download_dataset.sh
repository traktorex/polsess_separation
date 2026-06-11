#!/bin/bash
# download_dataset.sh — Fetch and extract the PolSESS dataset for polsess_separation.
#
# Companion to setup.sh: setup.sh installs the project, this script provisions data.
# Idempotent — refuses to overwrite an existing extraction unless --force is passed.
#
# Default source is the maintainer's rclone gdrive remote, since the public Drive link
# trips Drive's anti-abuse limit on large files and won't work with gdown. To use rclone
# you need an "rclone config" with a remote named 'gdrive' that has access to the file
# (run `setup.sh --rclone` once to set it up).
#
# Usage:
#   ./download_dataset.sh                    # default: rclone from gdrive:polsess/...
#   POLSESS_URL=... ./download_dataset.sh    # override source (gdrive: / HTTP / local file)
#   ./download_dataset.sh --force            # re-download even if already extracted

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
DATASET_NAME="PolSESS_C_final_128_v2"
ARCHIVE_NAME="${DATASET_NAME}.zip"

# Default: rclone remote path. Requires an `rclone config` with a remote named 'gdrive'.
DEFAULT_URL="gdrive:polsess/${ARCHIVE_NAME}"

POLSESS_URL="${POLSESS_URL:-$DEFAULT_URL}"

# Archive unzips to a single-level directory (no double-nesting).
EXTRACTED_PATH="$DATASETS_DIR/$DATASET_NAME"

# ============================================================================
# Helpers (matches setup.sh)
# ============================================================================

info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERR ]\033[0m $*"; exit 1; }

# ============================================================================
# Parse arguments
# ============================================================================

FORCE=false
while [ $# -gt 0 ]; do
    case "$1" in
        --force)
            FORCE=true; shift ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            error "Unknown argument: $1" ;;
    esac
done

# ============================================================================
# Skip if already present
# ============================================================================

if [ -d "$EXTRACTED_PATH" ] && [ "$FORCE" = false ]; then
    ok "Dataset already present at $EXTRACTED_PATH"
    info "Pass --force to re-download"
    exit 0
fi

# ============================================================================
# Ensure unzip is available (the archive is a .zip — fails noisily without it)
# ============================================================================

if ! command -v unzip &>/dev/null; then
    info "Installing unzip..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq unzip
    else
        error "unzip not found and apt-get unavailable — install unzip manually"
    fi
fi

# ============================================================================
# Download
# ============================================================================

mkdir -p "$DATASETS_DIR"
archive="/tmp/${ARCHIVE_NAME}"

info "Downloading $DATASET_NAME from: $POLSESS_URL"

if [[ "$POLSESS_URL" == *":"* && "$POLSESS_URL" != http* ]]; then
    # rclone remote path (e.g. gdrive:polsess/foo.zip)
    if ! command -v rclone &>/dev/null; then
        error "rclone not installed — run './setup.sh --rclone' first, or override POLSESS_URL with an HTTP URL or local file"
    fi
    rclone copy "$POLSESS_URL" /tmp/ --progress
    src_name="$(basename "$POLSESS_URL")"
    [ -f "/tmp/$src_name" ] && [ "/tmp/$src_name" != "$archive" ] && mv "/tmp/$src_name" "$archive"
elif [[ "$POLSESS_URL" == *"drive.google.com"* ]]; then
    # Last-resort path — Drive's anti-abuse usually kills this for the 47 GB archive.
    warn "Using gdown against drive.google.com — large public files often hit Drive's quota wall. Prefer rclone."
    pip install --upgrade --no-cache-dir 'gdown>=6.0.0'
    gdrive_id="$(echo "$POLSESS_URL" | sed -nE 's#.*[?&]id=([^&]+).*#\1#p; s#.*/file/d/([^/]+)/.*#\1#p' | head -1)"
    [ -z "$gdrive_id" ] && error "Could not extract Drive file ID from $POLSESS_URL"
    python3 -c "import gdown; gdown.download(id='$gdrive_id', output='$archive', quiet=False)"
elif [ -f "$POLSESS_URL" ]; then
    cp "$POLSESS_URL" "$archive"
else
    wget -q --show-progress -O "$archive" "$POLSESS_URL"
fi

ok "Archive downloaded ($(du -h "$archive" | cut -f1))"

# ============================================================================
# Extract
# ============================================================================

info "Extracting to $DATASETS_DIR..."
unzip -q "$archive" -d "$DATASETS_DIR"
rm -f "$archive"
ok "Extraction complete"

if [ ! -d "$EXTRACTED_PATH" ]; then
    error "Expected path missing after extraction: $EXTRACTED_PATH (archive layout may differ — check $DATASETS_DIR)"
fi

# ============================================================================
# Wire env var into ~/.bashrc
# ============================================================================

if grep -q "^export POLSESS_DATA_ROOT=" ~/.bashrc 2>/dev/null; then
    current="$(grep "^export POLSESS_DATA_ROOT=" ~/.bashrc | tail -1)"
    if [[ "$current" == *"$EXTRACTED_PATH\""* ]]; then
        ok "POLSESS_DATA_ROOT already pointing at $EXTRACTED_PATH"
    else
        warn "POLSESS_DATA_ROOT is set in ~/.bashrc but to a different path:"
        warn "  $current"
        warn "  Edit ~/.bashrc manually if you want to switch to $EXTRACTED_PATH"
    fi
else
    info "Adding POLSESS_DATA_ROOT to ~/.bashrc..."
    cat >> ~/.bashrc << ENVEOF

# polsess_separation dataset (added by download_dataset.sh)
export POLSESS_DATA_ROOT="$EXTRACTED_PATH"
ENVEOF
    ok "Env var persisted — run 'source ~/.bashrc' to load it in this shell"
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo "============================================"
ok "Dataset ready at $EXTRACTED_PATH"
echo "============================================"
echo ""
info "Top-level contents:"
ls "$EXTRACTED_PATH" | sed 's/^/  /'
echo ""
echo "Next:"
echo "  source ~/.bashrc"
echo "  cd ~/polsess_separation && python train.py --config experiments/sepformer/sepformer_baseline.yaml"
echo ""
