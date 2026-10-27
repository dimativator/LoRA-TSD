#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GENERATION_ROOT="${GENERATION_ROOT:-/home/jovyan/shares/SR006.nfs3/dimativator/lora-generation-1b-20261026}"
export HF_HOME=/home/jovyan/.cache/huggingface
export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
python -c 'import sys, pathlib; sys.path.insert(0, sys.argv[1]); from disk_guard import DiskGuard; DiskGuard(pathlib.Path(sys.argv[2]), "bootstrap")(pending_write_bytes=3000000000, reason="before_runtime_setup", write_path=pathlib.Path("/tmp"))' "$ROOT/scripts/generation" "$GENERATION_ROOT"
# Restore the repository's declared versions without modifying the shared cache.
python -m pip install --no-cache-dir --no-deps --target /tmp/generation-extra peft==0.17.1 transformers==4.57.3 accelerate==1.10.1 tokenizers==0.22.2 huggingface-hub==0.36.0
export PYTHONPATH="/tmp/generation-extra${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${GENERATION_INSTALL_IFEVAL_DEPS:-0}" == "1" ]]; then
    python -m pip install --no-cache-dir --no-deps --target /tmp/generation-extra langdetect==1.0.9 immutabledict==4.2.2
fi
if [[ "${1:-}" == "cpu-smoke" ]]; then
    exec python -u "$ROOT/scripts/generation/cpu_smoke.py"
fi
if [[ "${1:-}" == "structured-smoke" ]]; then
    shift
    exec python -u "$ROOT/scripts/generation/structured_smoke.py" "$@"
fi
exec python -u "$ROOT/scripts/generation/supervise.py" "$@"
