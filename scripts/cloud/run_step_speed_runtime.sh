#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="${SPEED_MODEL_OWNED_ROOT:?}/hf-home"
export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
python -c 'import sys,pathlib; sys.path.insert(0,sys.argv[1]); from disk_guard import DiskGuard; DiskGuard(pathlib.Path(sys.argv[2]),"bootstrap")(pending_write_bytes=1000000000,reason="before_runtime_setup",write_path=pathlib.Path("/tmp"))' "$ROOT/scripts/generation" "$SPEED_ROOT"
# Match the versions already declared and used by the existing research jobs.
python -m pip install --no-cache-dir --no-deps --target /tmp/step-speed-extra peft==0.17.1 transformers==4.57.3 accelerate==1.10.1 tokenizers==0.22.2 huggingface-hub==0.36.0
export PYTHONPATH="/tmp/step-speed-extra${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "preflight" ]]; then
    exec python -u "$ROOT/scripts/step_speed/preflight.py"
else
    exec python -u "$ROOT/scripts/step_speed/supervise.py" "$@"
fi
