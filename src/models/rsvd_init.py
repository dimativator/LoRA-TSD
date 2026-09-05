"""Gradient-aligned initialization for standard and Riemannian LoRA layers."""

import torch
from .riemannian_lora_layer import RiemannianLoRALinear


@torch.no_grad()
def _zero_B_eff(mod: RiemannianLoRALinear):
    """Zero the effective B factor (first r columns of lora_B.weight)."""
    mod.lora_B.weight.data[:, :mod.r] = 0.0


@torch.no_grad()
def _zero_A_eff(mod: RiemannianLoRALinear):
    """Zero the effective A factor (first r rows of lora_A.weight)."""
    mod.lora_A.weight.data[:mod.r, :] = 0.0


@torch.no_grad()
def _get_grad_B_eff(mod: RiemannianLoRALinear) -> torch.Tensor:
    """Return the gradient of the active B factor."""
    r = mod.r
    g = mod.lora_B.weight.grad
    if g is None:
        return None
    return g[:, :r].T.clone()


@torch.no_grad()
def _get_grad_A_eff(mod: RiemannianLoRALinear) -> torch.Tensor:
    """Return the gradient of the active A factor."""
    r = mod.r
    g = mod.lora_A.weight.grad
    if g is None:
        return None
    return g[:r, :].T.clone()


@torch.no_grad()
def _set_A_eff_orth(mod: RiemannianLoRALinear, grad_A: torch.Tensor):
    """Set the active A factor from a QR basis and zero B."""
    r = mod.r
    if grad_A is None or grad_A.norm() < 1e-12:
        Q, _ = torch.linalg.qr(torch.randn_like(grad_A))
    else:
        Q, _ = torch.linalg.qr(grad_A)
    Q = Q / (Q.shape[0] ** 0.5)

    mod.lora_A.weight.data.zero_()
    mod.lora_A.weight.data[:r, :] = Q.T
    mod.lora_B.weight.data.zero_()


@torch.no_grad()
def _set_B_eff_orth(mod: RiemannianLoRALinear, grad_B: torch.Tensor):
    """Set the active B factor from a QR basis and zero A."""
    r = mod.r
    if grad_B is None or grad_B.norm() < 1e-12:
        Q, _ = torch.linalg.qr(torch.randn(grad_B.shape[1], r,
                                            dtype=grad_B.dtype, device=grad_B.device))
    else:
        Q, _ = torch.linalg.qr(grad_B.T)
    Q = Q / (Q.shape[0] ** 0.5)

    mod.lora_B.weight.data.zero_()
    mod.lora_B.weight.data[:, :r] = Q
    mod.lora_A.weight.data[:r, :] = 0.0


def rsvd_init(model, dataloader, q: int = 2, device: str = "cuda", scale: float = 0.0):
    """Initialize Riemannian LoRA layers using 2*q + 1 training batches."""
    riem_modules = [m for m in model.modules() if isinstance(m, RiemannianLoRALinear)]
    if not riem_modules:
        print("[rsvd_init] No RiemannianLoRALinear modules found - skipping.")
        return

    print(f"[rsvd_init] Initializing {len(riem_modules)} modules, q={q}, scale={scale}")
    model.train()
    data_iter = iter(dataloader)

    def _forward_backward():
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()

    # Alternating QR subspace iterations.
    for _ in range(q):
        for mod in riem_modules:
            _zero_B_eff(mod)

        _forward_backward()

        for mod in riem_modules:
            grad_B = _get_grad_B_eff(mod)
            _set_B_eff_orth(mod, grad_B)

        _forward_backward()

        for mod in riem_modules:
            grad_A = _get_grad_A_eff(mod)
            _set_A_eff_orth(mod, grad_A)

    # Final SVD pass.
    for mod in riem_modules:
        _zero_B_eff(mod)

    _forward_backward()

    with torch.no_grad():
        for mod in riem_modules:
            r = mod.r

            grad_B = _get_grad_B_eff(mod)
            A_eff = mod.lora_A.weight.data[:r, :].T

            if grad_B is None or grad_B.norm() < 1e-12:
                print(f"[rsvd_init] Warning: zero gradient for module {mod}, skipping SVD step.")
                continue

            A_eff_orth, _ = torch.linalg.qr(A_eff)

            U, S, Vt = torch.linalg.svd(grad_B, full_matrices=False)
            A_eff_new = A_eff_orth @ U
            B_eff_new = Vt

            mod.lora_A.weight.data.zero_()
            mod.lora_A.weight.data[:r, :] = A_eff_new.T

            mod.lora_B.weight.data.zero_()
            mod.lora_B.weight.data[:, :r] = (scale * B_eff_new).T
            B_orth, _ = torch.linalg.qr(B_eff_new.T)
            mod.lora_B.weight.data[:, r:] = B_orth

    model.zero_grad()
    print("[rsvd_init] Done.")


def rsvd_init_standard_lora(model, dataloader, q: int = 2, device: str = "cuda", scale: float = 0.0):
    """Initialize standard PEFT LoRA layers using 2*q + 1 training batches."""
    lora_pairs = []
    for module in model.modules():
        if isinstance(module, RiemannianLoRALinear):
            continue
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for adapter_name, A_layer in module.lora_A.items():
                B_layer = module.lora_B[adapter_name]
                lora_pairs.append((A_layer.weight, B_layer.weight))

    if not lora_pairs:
        print("[rsvd_init_standard_lora] No standard PEFT LoRA modules found - skipping.")
        return

    print(f"[rsvd_init_standard_lora] Initializing {len(lora_pairs)} modules, q={q}, scale={scale}")
    model.train()
    data_iter = iter(dataloader)

    def _forward_backward():
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()

    with torch.no_grad():
        for A, B in lora_pairs:
            r, in_features = A.shape
            Q, _ = torch.linalg.qr(torch.randn(in_features, r, dtype=A.dtype, device=A.device))
            A.data.copy_(Q.T)
            B.data.zero_()

    # Alternating QR subspace iterations.
    for _ in range(q):
        with torch.no_grad():
            for A, B in lora_pairs:
                B.data.zero_()

        _forward_backward()

        with torch.no_grad():
            for A, B in lora_pairs:
                if B.grad is None:
                    continue
                grad_B = B.grad.clone()
                Q, _ = torch.linalg.qr(grad_B)
                B.data.copy_(Q / (Q.shape[0] ** 0.5))
                A.data.zero_()

        _forward_backward()

        with torch.no_grad():
            for A, B in lora_pairs:
                if A.grad is None:
                    continue
                grad_A = A.grad.clone()
                Q, _ = torch.linalg.qr(grad_A.T)
                A.data.copy_((Q / (Q.shape[0] ** 0.5)).T)
                B.data.zero_()

    # Final SVD pass.
    with torch.no_grad():
        for A, B in lora_pairs:
            B.data.zero_()

    _forward_backward()

    with torch.no_grad():
        for A, B in lora_pairs:
            if B.grad is None:
                print("[rsvd_init_standard_lora] Warning: zero gradient, skipping SVD step.")
                continue

            grad_B = B.grad.clone()
            A_current = A.data.clone()

            if grad_B.norm() < 1e-12:
                print("[rsvd_init_standard_lora] Warning: near-zero gradient, skipping.")
                continue

            Q, _ = torch.linalg.qr(A_current.T)
            A_orth = Q.T

            U, S, Vt = torch.linalg.svd(grad_B, full_matrices=False)
            B.data.copy_(scale * U)
            A.data.copy_(Vt @ A_orth)

    model.zero_grad()
    print("[rsvd_init_standard_lora] Done.")
