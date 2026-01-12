#!/usr/bin/env bash
set -euo pipefail

# Go to repo root (works no matter where you run it from)
cd "$(dirname "$0")/.."

# Activate your napari venv
source ~/venvs/napari-311/bin/activate

# Make sure "vesuvius" imports resolve
export PYTHONPATH="$PWD/src"

# Run the viewer
python scripts/view_napari_volume.py \
  --sid "${1:?Usage: scripts/napari_view.sh <sid>}" \
  "${@:2}"