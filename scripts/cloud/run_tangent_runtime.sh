#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python -m pip install --no-cache-dir --no-deps --target /tmp/tangent-extra peft==0.17.1 transformers==4.57.3 accelerate==1.10.1 tokenizers==0.22.2 huggingface-hub==0.36.0
export PYTHONPATH="/tmp/tangent-extra${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "preflight" ]]; then
    exec python -u "$ROOT/scripts/tangent_ablation/preflight.py"
elif [[ "${1:-}" == "smoke" ]]; then
    exec python -u "$ROOT/scripts/tangent_ablation/gpu_smoke.py"
else
    exec python -u "$ROOT/scripts/tangent_ablation/supervise.py" "$@"
fi
