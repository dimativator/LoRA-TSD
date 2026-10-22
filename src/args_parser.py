import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--optimizer",
        required=True,
        type=lambda value: "riemannion_original" if value.lower() == "riemannionoriginal" else value.lower(),
        choices=[
            "lora_sgd",
            "adamw",
            "lora_muon",
            "riemannian_sgd",
            "lora_rite",
            "riemannion",
            "riemannion_original",
            "lora_pro",
            "lora_muon_manifold",
            "lora_tsd",
            "muon_full",
        ],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=600)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from", default=None)

    parser.add_argument("--lr-a", type=float, required=True)
    parser.add_argument("--lr-b", type=float, default=5e-3)
    parser.add_argument("--lr-scheduler", choices=["linear"], default=None)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--retraction", choices=["native", "factor", "svd", "orthographic", "ksl"], default="native")
    parser.add_argument("--retraction-gauge", choices=["native", "left_orthogonal"], default="native")
    parser.add_argument("--retraction-momentum", choices=["factor", "projected"], default="factor")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-delta-norm", type=float, default=0.5)

    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lora-pro-eps-inv", type=float, default=1e-6)

    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-nesterov", action="store_true", default=True)
    parser.add_argument("--muon-no-nesterov", dest="muon_nesterov", action="store_false")

    parser.add_argument("--riemannian-momentum", type=float, default=0.95)
    parser.add_argument("--riemannion-nesterov", action="store_true")
    parser.add_argument("--riemannion-nuclear-scale", action="store_true")

    parser.add_argument("--lora-rite-beta1", type=float, default=0.9)
    parser.add_argument("--lora-rite-beta2", type=float, default=0.999)
    parser.add_argument("--lora-rite-eps", type=float, default=1e-6)
    parser.add_argument("--lora-rite-clip", type=float, default=1.0)
    parser.add_argument("--lora-rite-cap", type=float, default=0.0)
    parser.add_argument("--lora-rite-skip", type=float, default=1.0)
    parser.add_argument("--lora-rite-escape", action="store_true")
    parser.add_argument("--lora-rite-balance", action="store_true")

    parser.add_argument("--lmm-momentum", type=float, default=0.95)
    parser.add_argument("--lmm-ns-steps", type=int, default=8)
    parser.add_argument("--lmm-inv-sqrt-steps", type=int, default=7)
    parser.add_argument("--lmm-eps", type=float, default=1e-5)
    parser.add_argument("--lmm-exact-inv-sqrt", action="store_true")

    parser.add_argument("--use-momentum", action="store_true")
    parser.add_argument("--momentum-cf", type=float, default=0.9)
    parser.add_argument("--loss-alpha", type=float, default=None)
    parser.add_argument("--ball-iters", type=int, default=5)
    parser.add_argument("--ball-ridge-eps", type=float, default=1e-8)
    parser.add_argument("--ball-balance", action="store_true")
    parser.add_argument("--ball-balance-mode", choices=["norm", "whiten"], default="norm")
    parser.add_argument("--ball-rebalance-every", type=int, default=1)

    parser.add_argument("--rsvd-init", action="store_true")
    parser.add_argument("--rsvd-q", type=int, default=15)
    parser.add_argument("--rsvd-scale", type=float, default=0.0)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["k_proj", "q_proj", "v_proj"],
    )
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser


def parse_config(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    config = vars(args)
    aliases = {
        "lr_a": "lr_A",
        "lr_b": "lr_B",
        "loss_alpha": "alpha",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_target_modules": "lora_target_modules",
        "lora_dropout": "lora_dropout",
    }
    for source, target in aliases.items():
        config[target] = config.pop(source)
    config["lora_bias"] = "none"
    config["lora_init"] = None
    config["add_W"] = False
    config["lr_A_scheduler"] = None
    config["lr_B_scheduler"] = None
    return config
