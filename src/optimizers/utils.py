import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """Approximate the polar factor of G with a quintic Newton-Schulz iteration."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.no_grad()
def _topk_eigs_gram(X: torch.Tensor, k: int = 16, niter: int = 25, eps: float = 1e-12):
    """Return the leading eigenvalues and singular values of the smaller Gram matrix."""
    m, n = X.shape
    gram = (X @ X.T) if (m <= n) else (X.T @ X)
    gram = 0.5 * (gram + gram.T)

    d = gram.shape[0]
    if d == 0:
        return torch.empty(0, device=X.device), torch.empty(0, device=X.device)

    k_eff = int(min(k, max(d - 1, 1)))
    if d <= 64 or k_eff >= d:
        evals = torch.linalg.eigvalsh(gram)
        evals = torch.flip(evals, dims=(0,))
        evals = torch.clamp(evals, min=0.0)
        svals = torch.sqrt(evals + eps)
        return evals[:k], svals[:k]

    X0 = torch.randn(d, k_eff, device=gram.device, dtype=gram.dtype)
    evals, _ = torch.lobpcg(gram, k=k_eff, X=X0, niter=niter, largest=True)
    evals = torch.sort(evals, descending=True).values
    evals = torch.clamp(evals, min=0.0)
    svals = torch.sqrt(evals + eps)
    return evals, svals


@torch.no_grad()
def power_iteration_t(X: torch.Tensor, iters: int = 1, eps: float = 1e-8):
    _, n = X.shape

    v = torch.randn(n, device=X.device, dtype=X.dtype)
    v /= v.norm() + eps

    for _ in range(iters):
        u = X @ v
        u /= u.norm() + eps

        v = X.t() @ u
        v /= v.norm() + eps

    u = X @ v
    u /= u.norm() + eps

    sigma = (X @ v).norm() + eps
    return sigma, u, v


@torch.no_grad()
def top_singular_triplet(X: torch.Tensor, iters: int = 10, use_svd: bool = False):
    if use_svd:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        sigma = S[0]
        u = U[:, 0]
        v = Vh[0, :].transpose(0, 0)
        v = Vh[0, :]
        return sigma, u, v
    else:
        sigma, u, v = power_iteration_t(X, iters=iters)
        return sigma, u, v
    

@torch.no_grad()
def power_iteration_lora(
        A: torch.Tensor,
        B: torch.Tensor,
        dA: torch.Tensor,
        dB: torch.Tensor,
        iters: int = 1, 
        eps: float = 1e-8
    ):
    _, n = A.shape

    v = torch.randn(n, device=A.device, dtype=B.dtype)
    v /= v.norm() + eps

    for _ in range(iters):
        u = dB @ (A @ v) + B @ (dA @ v)
        u /= u.norm() + eps

        v = A.t() @ (dB.t() @ u) + dA.t() @ (B.t() @ u)
        v /= v.norm() + eps

    u = dB @ (A @ v) + B @ (dA @ v)
    sigma = u.norm() + eps
    u /= sigma
    
    return sigma, u, v


@torch.no_grad()
def lowrank_update_top_svd(
    A: torch.Tensor, 
    B: torch.Tensor, 
    dA: torch.Tensor,
    dB: torch.Tensor,
    k: int = None,
):
    P = torch.cat([dB, B], dim=1)
    Q = torch.cat([A, dA], dim=0)

    Up, Rp = torch.linalg.qr(P, mode="reduced")

    Vq, Rq = torch.linalg.qr(Q.T, mode="reduced")

    M = Rp @ Rq.T

    Um, S, Vmh = torch.linalg.svd(M, full_matrices=False)

    if k is None:
        k = S.numel()
    else:
        k = min(k, S.numel())

    Um = Um[:, :k]
    S = S[:k]
    Vmh = Vmh[:k, :]

    U = Up @ Um
    V = Vq @ Vmh.T

    return U, S, V


def top_singular_triplet_lora(
        A: torch.Tensor,
        B: torch.Tensor,
        dA: torch.Tensor,
        dB: torch.Tensor,
        iters: int = 10, 
        use_svd: bool = False,
        num_vectors: int = 1,
    ):
    if use_svd:
        U, S, V = lowrank_update_top_svd(A, B, dA, dB, k=num_vectors)
        sigma = S[0]
        return sigma, U, V
    else:
        sigma, u, v = power_iteration_lora(A, B, dA, dB, iters=iters)
        return sigma, u, v
