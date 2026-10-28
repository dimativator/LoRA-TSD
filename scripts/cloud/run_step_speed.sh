#!/usr/bin/env bash
set -euo pipefail
trap 'exit_code=$?; echo "SPEED_ENTRY_EXIT=$exit_code"; exit 0' EXIT
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
python -u "$ROOT/scripts/step_speed/model_storage.py" "$@"
