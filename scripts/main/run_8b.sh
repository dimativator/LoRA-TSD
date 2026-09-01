#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B}"
DATASETS=(${DATASETS:-boolq piqa siqa obqa qnli multi_nli})
SEED="${SEED:-42}"

for dataset in "${DATASETS[@]}"; do
  python "${ROOT}/src/main.py" \
    --run-name "llama31_8b_${dataset}_seed${SEED}" \
    --optimizer lora_tsd \
    --model "${MODEL}" \
    --dataset "${dataset}" \
    --output-dir "${ROOT}/results/main_8b" \
    --seed "${SEED}" \
    --steps 3000 \
    --train-batch-size 16 \
    --grad-accum-steps 2 \
    --eval-every 250 \
    --grad-clip-norm 1 \
    --lr-a 4e-3 \
    --lr-b 4e-3 \
    --lr-scheduler linear \
    --max-delta-norm 0.1 \
    --use-momentum \
    --momentum-cf 0.95 \
    --ball-iters 5 \
    --ball-balance \
    --ball-balance-mode norm \
    --ball-rebalance-every 250 \
    --muon-ns-steps 5 \
    --rsvd-init \
    --rsvd-q 15 \
    --rsvd-scale 0 \
    --lora-r 16 \
    --lora-alpha 16
done
