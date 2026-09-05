# LoRA-TSD

Code accompanying the anonymous submission
"LoRA-TSD: Tangent-Space Spectral Descent for LoRA via Muon-Style Updates."

This package contains LoRA-TSD, the comparison optimizers used in the paper,
and the scripts needed to reproduce the main experiments and ablations. Models
and datasets are loaded from Hugging Face. Results are written to local JSON
files, and the code does not use external experiment logging.

## Environment

The experiments were run with the `lora_llama` Conda environment. Its relevant
package versions are recorded in `environment.yaml`.

```bash
conda env create -f environment.yaml
conda activate lora_llama
```

The Llama checkpoints are gated. Authenticate with Hugging Face or set
`HF_TOKEN` before running the scripts.

## Experiments

Run commands from the package root:

```bash
bash scripts/main/run_1b.sh
bash scripts/main/run_8b.sh
bash scripts/main/run_all_1b_boolq.sh
bash scripts/tau_ablation/run.sh
bash scripts/rank_ablation/run.sh
bash scripts/qwen32b/run_siqa_obqa_fp8.sh
```

`run_1b.sh` and `run_8b.sh` evaluate LoRA-TSD on BoolQ, PIQA, SIQA, OpenBookQA,
QNLI, and MultiNLI. `run_all_1b_boolq.sh` runs every method in the 1B table on
BoolQ with the reported hyperparameters:

- SGD and AdamW applied to the LoRA factors
- Muon applied independently to the LoRA factors
- Riemannian SGD
- LoRA-Rite
- Riemannion
- LoRA-Pro
- LoRA-Muon
- LoRA-TSD
- Muon full fine-tuning

The 1B runs use seeds 42, 0, and 1. The ablation scripts reproduce the inner-iteration sweep on the 1B model and the rank sweep on the 8B model.

The lists of datasets, seeds, inner iterations, and ranks can be overridden
through environment variables. For example:

```bash
DATASETS="boolq siqa" SEEDS="42" bash scripts/main/run_1b.sh
METHODS="lora_pro lora_tsd" SEEDS="42" bash scripts/main/run_all_1b_boolq.sh
TAU_VALUES="1 3 5" bash scripts/tau_ablation/run.sh
RANKS="8 16 32" bash scripts/rank_ablation/run.sh
```

Each run stores a JSON file under `results/`. `reported_accuracy` is the maximum
of the final evaluation and the evaluations performed during training, matching
the table construction. The file also records the individual evaluations, a
loss summary, and average training time per step.

## Qwen3-32B FP8 experiments

The base model is loaded as frozen TorchAO FP8 weight-only weights. LoRA factors and optimizer state remain in FP32. The comparison contains AdamW, factor-wise Muon (`lora_muon`), LoRA-Muon Manifold, LoRA-Rite, and LoRA-TSD.

Run the complete matrix:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/qwen32b/run_siqa_obqa_fp8.sh
```

The script records the exact microbatch/accumulation and optimizer-specific settings used for each reported configuration. Override `DATASETS` or `METHODS` to run a subset:

```bash
DATASETS="siqa" METHODS="lora_tsd" \
  CUDA_VISIBLE_DEVICES=0 bash scripts/qwen32b/run_siqa_obqa_fp8.sh
```

Rolling checkpoints are saved atomically every 400 optimizer updates under
`checkpoints/qwen3_32b_fp8/`. Re-running the same command resumes from
`latest.pt`, including LoRA parameters, optimizer state, RNG state, dataloader
position, losses, and validation scores. The frozen 32B base weights are not
duplicated in the checkpoint. Checkpoints are retained after successful runs.
