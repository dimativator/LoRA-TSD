#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; echo "ABLATION_WRAPPER_EXIT=$code"; exit 0' EXIT
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TMPDIR=/tmp
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_DATASETS_CACHE=/tmp/tangent-data/hf-cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
python -u "$ROOT/scripts/tangent_ablation/bootstrap.py" "$@"
