#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
DATASETS=(${DATASETS:-boolq piqa siqa})
TAU_VALUES=(${TAU_VALUES:-1 2 3 4 5})
SEED="${SEED:-42}"

for dataset in "${DATASETS[@]}"; do
  for tau in "${TAU_VALUES[@]}"; do
    python "${ROOT}/src/main.py" \
      --run-name "tau_${tau}_${dataset}_seed${SEED}" \
      --optimizer lora_tsd \
      --model "${MODEL}" \
      --dataset "${dataset}" \
      --output-dir "${ROOT}/results/tau_ablation" \
      --seed "${SEED}" \
      --steps 3000 \
      --train-batch-size 16 \
      --grad-accum-steps 2 \
      --eval-every 600 \
      --grad-clip-norm 1 \
      --lr-a 5e-3 \
      --lr-b 5e-3 \
      --lr-scheduler linear \
      --max-delta-norm 0.1 \
      --use-momentum \
      --momentum-cf 0.95 \
      --ball-iters "${tau}" \
      --ball-balance \
      --ball-balance-mode norm \
      --ball-rebalance-every 1 \
      --muon-ns-steps 5 \
      --rsvd-init \
      --rsvd-q 15 \
      --rsvd-scale 0 \
      --lora-r 16 \
      --lora-alpha 16
  done
done
