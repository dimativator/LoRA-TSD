#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
SEEDS=(${SEEDS:-42 0 1})
METHODS=(${METHODS:-sgd adamw muon riem_sgd lora_rite riemannion lora_pro lora_muon lora_tsd muon_ft})

run_method() {
  local method="$1"
  local seed="$2"
  local common=(
    --run-name "llama32_1b_boolq_${method}_seed${seed}"
    --model "${MODEL}"
    --dataset boolq
    --output-dir "${ROOT}/results/main_1b_boolq"
    --seed "${seed}"
    --lora-r 16
    --lora-alpha 16
  )

  case "${method}" in
    sgd)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer lora_sgd --steps 3000 --train-batch-size 32 \
        --grad-accum-steps 1 --eval-every 600 --lr-a 1e-3 --lr-b 1e-3 \
        --use-momentum --momentum-cf 0.9 --lr-scheduler linear \
        --rsvd-init --rsvd-q 15 --rsvd-scale 0
      ;;
    adamw)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer adamw --steps 3000 --train-batch-size 32 \
        --grad-accum-steps 1 --eval-every 600 --grad-clip-norm 1 \
        --lr-a 1e-3 --lr-b 1e-3 --lr-scheduler linear \
        --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 \
        --adam-eps 1e-8 --max-delta-norm 0.5 \
        --rsvd-init --rsvd-q 15 --rsvd-scale 0
      ;;
    muon)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer lora_muon --steps 3000 --train-batch-size 32 \
        --grad-accum-steps 1 --eval-every 600 --grad-clip-norm 1 \
        --lr-a 0.01 --lr-b 0.01 --lr-scheduler linear \
        --muon-momentum 0.95 --muon-ns-steps 5 --max-delta-norm 0.01 \
        --rsvd-init --rsvd-q 15 --rsvd-scale 0
      ;;
    riem_sgd)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer riemannian_sgd --steps 3000 --train-batch-size 32 \
        --grad-accum-steps 1 --eval-every 250 \
        --lr-a 0.002 --lr-b 0.005 --lr-scheduler linear \
        --riemannian-momentum 0.95 \
        --rsvd-init --rsvd-q 15 --rsvd-scale 0
      ;;
    lora_rite)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer lora_rite --steps 3000 --train-batch-size 16 \
        --grad-accum-steps 2 --eval-every 600 --grad-clip-norm 1 \
        --lr-a 8e-5 --lr-b 0.005 \
        --lora-rite-beta1 0.9 --lora-rite-beta2 0.999 \
        --lora-rite-eps 1e-6 --lora-rite-clip 1 \
        --rsvd-init --rsvd-q 15 --rsvd-scale 0
      ;;
    riemannion)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer riemannion --steps 3000 --train-batch-size 16 \
        --grad-accum-steps 2 --eval-every 600 \
        --lr-a 3e-5 --lr-b 0.005 --lr-scheduler linear \
        --riemannian-momentum 0.95 \
        --rsvd-init --rsvd-q 15 --rsvd-scale 0
      ;;
    lora_pro)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer lora_pro --steps 3000 --train-batch-size 16 \
        --grad-accum-steps 2 --eval-every 600 \
        --lr-a 5e-4 --lr-b 0.005 --weight-decay 0.1 \
        --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-6 \
        --lora-pro-eps-inv 1e-6 --max-delta-norm 0.1
      ;;
    lora_muon)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer lora_muon_manifold --steps 3000 --train-batch-size 32 \
        --grad-accum-steps 1 --eval-every 500 --grad-clip-norm 1 \
        --lr-a 0.003 --lr-b 0.003 --weight-decay 0.1 \
        --lmm-momentum 0.95 --lmm-ns-steps 8 \
        --lmm-inv-sqrt-steps 7 --lmm-eps 1e-5 --max-delta-norm 0.1 \
        --rsvd-init --rsvd-q 15 --rsvd-scale 0
      ;;
    lora_tsd)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer lora_tsd --steps 3000 --train-batch-size 16 \
        --grad-accum-steps 2 --eval-every 600 --grad-clip-norm 1 \
        --lr-a 5e-3 --lr-b 5e-3 --lr-scheduler linear \
        --max-delta-norm 0.1 --use-momentum --momentum-cf 0.95 \
        --ball-iters 3 \
        --ball-balance --ball-balance-mode norm --ball-rebalance-every 1 \
        --muon-ns-steps 5 --rsvd-init --rsvd-q 15 --rsvd-scale 0
      ;;
    muon_ft)
      python "${ROOT}/src/main.py" "${common[@]}" \
        --optimizer muon_full --steps 3000 --train-batch-size 8 \
        --grad-accum-steps 4 --eval-every 250 \
        --lr-a 0.03 --lr-b 0.005 --muon-momentum 0.95 \
        --muon-ns-steps 5 --max-delta-norm 0.5
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 2
      ;;
  esac
}

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_method "${method}" "${seed}"
  done
done
