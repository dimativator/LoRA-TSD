from .lora_adam import LoRAAdam
from .lora_muon import LoRAMuon
from .lora_muon_manifold import LoRAMuonManifold
from .lora_pro import LoRAPro
from .lora_rite import LoRARite
from .muon import Muon
from .riemannian_lora import RiemannianSGD, Riemannion
from .sgd import LoRASGD
from .lora_tsd import LoRATSD


def build_optimizer(params, config):
    name = config["optimizer"].lower()

    if name == "lora_sgd":
        return LoRASGD(
            params,
            lr_A=config["lr_A"],
            lr_B=config["lr_B"],
            max_delta_norm=config["max_delta_norm"],
            log_ns_gap=False,
        )

    if name == "adamw":
        return LoRAAdam(
            params,
            lr_A=config["lr_A"],
            lr_B=config["lr_B"],
            betas=(config["adam_beta1"], config["adam_beta2"]),
            eps=config["adam_eps"],
            weight_decay=config["weight_decay"],
            max_delta_norm=config["max_delta_norm"],
            log_ns_gap=False,
            log_optimizer_stats=False,
        )

    if name == "lora_muon":
        return LoRAMuon(
            params=params,
            lr_A=config["lr_A"],
            lr_B=config["lr_B"],
            momentum=config["muon_momentum"],
            nesterov=config["muon_nesterov"],
            ns_steps=config["muon_ns_steps"],
            max_delta_norm=config["max_delta_norm"],
            log_ns_gap=False,
            log_optimizer_stats=False,
        )

    if name == "lora_muon_manifold":
        return LoRAMuonManifold(
            params=params,
            lr=config["lr_A"],
            momentum=config["lmm_momentum"],
            weight_decay=config["weight_decay"],
            ns_steps=config["lmm_ns_steps"],
            inv_sqrt_steps=config["lmm_inv_sqrt_steps"],
            eps=config["lmm_eps"],
            exact_inv_sqrt=config["lmm_exact_inv_sqrt"],
            max_delta_norm=config["max_delta_norm"],
            log_optimizer_stats=False,
        )

    if name == "riemannian_sgd":
        return RiemannianSGD(
            params,
            lr=config["lr_A"],
            momentum=config["riemannian_momentum"],
        )

    if name == "riemannion":
        return Riemannion(
            params,
            lr=config["lr_A"],
            momentum=config["riemannian_momentum"],
            nesterov=config["riemannion_nesterov"],
            nuclear_scale=config["riemannion_nuclear_scale"],
            weight_decay=config["weight_decay"],
        )

    if name == "lora_rite":
        return LoRARite(
            params,
            betas=(config["lora_rite_beta1"], config["lora_rite_beta2"]),
            eps=config["lora_rite_eps"],
            lr=config["lr_A"],
            clip_unmagnified_grad=config["lora_rite_clip"],
            update_capping=config["lora_rite_cap"],
            update_skipping=config["lora_rite_skip"],
            apply_escape=config["lora_rite_escape"],
            balance_param=config["lora_rite_balance"],
            weight_decay=config["weight_decay"],
            lora_l_dim=0,
            lora_r_dim=-1,
        )

    if name == "lora_pro":
        return LoRAPro(
            params,
            lr=config["lr_A"],
            betas=(config["adam_beta1"], config["adam_beta2"]),
            eps_adam=config["adam_eps"],
            eps_inv=config["lora_pro_eps_inv"],
            weight_decay=config["weight_decay"],
            max_delta_norm=config["max_delta_norm"],
            log_ns_gap=False,
            log_optimizer_stats=False,
        )

    if name == "lora_tsd":
        return LoRATSD(
            params,
            norm_ord=2,
            use_momentum=config["use_momentum"],
            momentum_cf=config["momentum_cf"],
            lr_A=config["lr_A"],
            lr_B=config["lr_B"],
            lr_A_scheduler=config["lr_A_scheduler"],
            lr_B_scheduler=config["lr_B_scheduler"],
            alpha=config["alpha"],
            lambda_l2_A=0.0,
            lambda_l2_B=0.0,
            ball_iters=config["ball_iters"],
            ridge_eps=config["ball_ridge_eps"],
            max_delta_norm=config["max_delta_norm"],
            ns_steps=config["muon_ns_steps"],
            balance=config["ball_balance"],
            balance_mode=config["ball_balance_mode"],
            rebalance_every=config["ball_rebalance_every"],
            log_optimizer_stats=False,
        )

    if name == "muon_full":
        return Muon(
            muon_params=params,
            lr=config["lr_A"],
            momentum=config["muon_momentum"],
            nesterov=config["muon_nesterov"],
            ns_steps=config["muon_ns_steps"],
            max_delta_norm=config["max_delta_norm"],
        )

    raise ValueError(f"Unsupported optimizer '{config['optimizer']}'")
