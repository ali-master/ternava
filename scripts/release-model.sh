#!/usr/bin/env bash
# Attach the current model.bin to a GitHub Release as a versioned asset.
#
# Usage:
#   bash scripts/release-model.sh v0.1.0 path/to/model.bin
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <tag> <path-to-model.bin>"
    exit 1
fi

TAG="$1"
BIN_PATH="$2"

if [[ ! -f "$BIN_PATH" ]]; then
    echo "ERROR: $BIN_PATH not found"
    exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh CLI not installed"
    exit 1
fi

echo "Attaching $BIN_PATH to release $TAG..."
gh release create "$TAG" "$BIN_PATH" \
    --title "$TAG" \
    --notes "Model binary for Ternava $TAG. See README and eval/REPORT.md for quality + perf scorecard." \
    || gh release upload "$TAG" "$BIN_PATH"

echo "Done."