#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=python
if ! python -c 'import cryptography' >/dev/null 2>&1; then
    python -m venv --system-site-packages /tmp/siqa-decrypt-venv
    PYTHON=/tmp/siqa-decrypt-venv/bin/python
    "${PYTHON}" -m pip install --no-cache-dir cryptography
fi
"${PYTHON}" "${ROOT}/scripts/cloud/unpack_siqa.py"
unset LORA_SIQA_BUNDLE_KEY
bash /tmp/private-siqa-runtime/src/configs/cloud/run_siqa_retractions.sh "$@"
