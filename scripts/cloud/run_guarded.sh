#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/tmp/lora-siqa-hf
export SIQA_DATA_DIR=/tmp/lora-siqa-data
export HF_DATASETS_CACHE=/tmp/lora-siqa-datasets
export XDG_CACHE_HOME=/tmp/lora-siqa-cache
export MPLCONFIGDIR=/tmp/lora-siqa-matplotlib
export PYTHONPYCACHEPREFIX=/tmp/lora-siqa-pycache
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${LORA_INSPECT:-0}" == "1" ]]; then
    exec python "${ROOT}/scripts/cloud/disk_supervisor.py" --campaign-root "${LORA_SIQA_LOG_ROOT}" --planned-temp-write-bytes 0 -- python "${ROOT}/scripts/cloud/inspect_benchmarks.py"
fi
exec python "${ROOT}/scripts/cloud/disk_supervisor.py" --campaign-root "${LORA_SIQA_LOG_ROOT}" -- bash "${ROOT}/scripts/cloud/run_siqa_encrypted.sh" "$@"
