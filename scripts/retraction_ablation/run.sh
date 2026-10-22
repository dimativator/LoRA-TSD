#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
DATASETS=(${DATASETS:-siqa})
SEEDS=(${SEEDS:-42 0 1})
RETRACTIONS=(${RETRACTIONS:-factor svd orthographic ksl})
METHODS=(${METHODS:-riemannion_original})
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/retraction_ablation}"

for method in "${METHODS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for retraction in "${RETRACTIONS[@]}"; do
        case "${method}" in
          riemannion_original)
            lr="${LR:-3e-5}"
            extra=(--riemannian-momentum 0.95 --retraction-gauge left_orthogonal)
            ;;
          lora_tsd)
            lr="${LR:-5e-3}"
            extra=(--use-momentum --momentum-cf 0.95 --ball-iters 5
                   --ball-balance --ball-balance-mode norm --ball-rebalance-every 1)
            if [[ "${retraction}" == native ]]; then
              extra+=(--retraction-gauge native --retraction-momentum factor)
            else
              extra+=(--retraction-gauge left_orthogonal --retraction-momentum projected)
            fi
            ;;
          *) echo "Unsupported ablation method: ${method}" >&2; exit 2 ;;
        esac
        run_name="${method}_${retraction}_${dataset}_seed${seed}"
        "${PYTHON:-python}" "${ROOT}/src/main.py" \
          --run-name "${run_name}" --optimizer "${method}" \
          --model "${MODEL}" --dataset "${dataset}" \
          --output-dir "${OUTPUT_DIR}" --metrics-jsonl "${OUTPUT_DIR}/${run_name}.jsonl" \
          --seed "${seed}" --steps "${STEPS:-3000}" --eval-every "${EVAL_EVERY:-300}" \
          --train-batch-size 2 --grad-accum-steps 16 --grad-checkpointing \
          --grad-clip-norm 1 --lr-a "${lr}" --lr-b 5e-3 \
          --lr-scheduler linear --min-lr 1e-5 --retraction "${retraction}" \
          --rsvd-init --rsvd-q 15 --rsvd-scale 0 --lora-r 16 --lora-alpha 16 \
          "${extra[@]}"
      done
    done
  done
done
