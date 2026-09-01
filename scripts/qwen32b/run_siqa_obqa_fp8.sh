#!/usr/bin/env bash
set -euo pipefail

# Qwen3-32B FP8 LoRA comparison reported for SIQA and OpenBookQA.
# Run from any directory. A dedicated H200-class GPU is recommended.

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PACKAGE_ROOT"

export QUANT="${QUANT:-fp8}"
export MAX_GPU_MEM="${MAX_GPU_MEM:-52GiB}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="Qwen/Qwen3-32B"
STEPS="${STEPS:-3000}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-400}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-400}"
DATASETS="${DATASETS:-siqa obqa}"
METHODS="${METHODS:-adamw lora_muon lora_muon_manifold lora_rite lora_tsd}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3_32b_fp8}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/qwen3_32b_fp8}"

set_batch_profile() {
    local dataset="$1"
    local method="$2"
    train_batch_size=32
    grad_accum_steps=1
    eval_batch_size=8

    if [[ "$method" == "lora_muon" ]]; then
        train_batch_size=8
        grad_accum_steps=4
        eval_batch_size=1
    elif [[ "$method" == "adamw" && "$dataset" == "obqa" ]]; then
        train_batch_size=8
        grad_accum_steps=4
        eval_batch_size=1
    elif [[ "$method" == "lora_tsd" && "$dataset" == "siqa" ]]; then
        train_batch_size=4
        grad_accum_steps=8
        eval_batch_size=1
    fi
}

set_optimizer_args() {
    local dataset="$1"
    local method="$2"
    case "$method" in
        adamw)
            optimizer_args=(
                --optimizer adamw
                --lr-a 1e-4 --lr-b 1e-4
                --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8
                --weight-decay 0.01 --lr-scheduler linear
                --max-delta-norm 0.01
            )
            ;;
        lora_muon)
            optimizer_args=(
                --optimizer lora_muon
                --lr-a 5e-4 --lr-b 5e-4
                --muon-momentum 0.95 --muon-ns-steps 5
                --weight-decay 0.0 --lr-scheduler linear
                --max-delta-norm 0.01
            )
            ;;
        lora_muon_manifold)
            optimizer_args=(
                --optimizer lora_muon_manifold
                --lr-a 3e-3 --lr-b 3e-3
                --lmm-momentum 0.95 --lmm-ns-steps 8
                --lmm-inv-sqrt-steps 7 --lmm-eps 1e-5
                --weight-decay 0.1 --max-delta-norm 0.1
            )
            ;;
        lora_rite)
            optimizer_args=(
                --optimizer lora_rite
                --lr-a 1e-4 --lr-b 1e-4
                --lora-rite-beta1 0.9 --lora-rite-beta2 0.999
                --lora-rite-eps 1e-6 --lora-rite-clip 1.0
                --weight-decay 0.0 --lr-scheduler linear
                --max-delta-norm 0.5
            )
            ;;
        lora_tsd)
            local lora_tsd_lr=3e-3
            if [[ "$dataset" == "siqa" ]]; then
                lora_tsd_lr=5e-3
            fi
            optimizer_args=(
                --optimizer lora_tsd
                --lr-a "$lora_tsd_lr" --lr-b "$lora_tsd_lr"
                --use-momentum --momentum-cf 0.95
                --muon-ns-steps 5 --ball-iters 3 --ball-balance
                --ball-balance-mode norm --ball-rebalance-every 250
                --weight-decay 0.0 --lr-scheduler linear
                --max-delta-norm 0.1
            )
            ;;
        *)
            echo "Unsupported optimizer: $method" >&2
            return 2
            ;;
    esac
}

for dataset in $DATASETS; do
    for method in $METHODS; do
        set_batch_profile "$dataset" "$method"
        set_optimizer_args "$dataset" "$method"
        run_name="qwen3_32b_fp8_${dataset}_${method}_seed${SEED}"
        checkpoint_dir="$CHECKPOINT_ROOT/$run_name"

        python -u src/main.py \
            --run-name "$run_name" \
            --model "$MODEL" \
            --dataset "$dataset" \
            --output-dir "$OUTPUT_DIR" \
            --steps "$STEPS" \
            --seed "$SEED" \
            --eval-every "$EVAL_EVERY" \
            --eval-batch-size "$eval_batch_size" \
            --train-batch-size "$train_batch_size" \
            --grad-accum-steps "$grad_accum_steps" \
            --grad-clip-norm 0.25 \
            --checkpoint-dir "$checkpoint_dir" \
            --checkpoint-every "$CHECKPOINT_EVERY" \
            --rsvd-init --rsvd-q 15 --rsvd-scale 0.0 \
            --lora-r 16 --lora-alpha 16 \
            --lora-target-modules q_proj k_proj v_proj \
            --lora-dropout 0.05 \
            "${optimizer_args[@]}"
    done
done
