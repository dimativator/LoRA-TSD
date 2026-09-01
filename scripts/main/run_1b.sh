#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
DATASETS=(${DATASETS:-boolq piqa siqa obqa qnli multi_nli})
SEEDS=(${SEEDS:-42 0 1})

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python "${ROOT}/src/main.py" \
      --run-name "llama32_1b_${dataset}_seed${seed}" \
      --optimizer lora_tsd \
      --model "${MODEL}" \
      --dataset "${dataset}" \
      --output-dir "${ROOT}/results/main_1b" \
      --seed "${seed}" \
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
      --ball-iters 5 \
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
