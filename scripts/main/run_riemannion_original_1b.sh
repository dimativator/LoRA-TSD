#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATASETS="${DATASETS:-boolq piqa siqa obqa qnli multi_nli}"
export METHODS=riemannion_original
export RETRACTIONS="${RETRACTIONS:-svd}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/riemannion_original_1b}"
bash "${ROOT}/scripts/retraction_ablation/run.sh"
