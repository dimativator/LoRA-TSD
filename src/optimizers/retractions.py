"""Fixed-rank retractions for W = B @ A, without dense weight matrices."""

import torch


RETRACTIONS = ("factor", "svd", "orthographic", "ksl")


@torch.no_grad()
def left_orthogonalize(A: torch.Tensor, B: torch.Tensor) -> tuple:
    Q, R = torch.linalg.qr(B, mode="reduced")
    return R @ A, Q


@torch.no_grad()
def truncated_svd(left: torch.Tensor, right: torch.Tensor, rank: int) -> tuple:
    Q_l, R_l = torch.linalg.qr(left, mode="reduced")
    Q_r, R_r = torch.linalg.qr(right.T, mode="reduced")
    U, S, Vt = torch.linalg.svd(R_l @ R_r.T, full_matrices=False)
    return S[:rank, None] * (Vt[:rank] @ Q_r.T), Q_l @ U[:, :rank]


@torch.no_grad()
def retract_candidate(
    A: torch.Tensor,
    B: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    method: str,
) -> tuple:
    """Retract Y=W+xi given Y=left@right. Return (A_new, B_new, fallback).

    At a rank-deficient starting point the fixed-rank tangent geometry is
    undefined. All non-SVD maps use the same SVD warm start there. Once on the
    manifold, a singular orthographic core raises rather than changing maps.
    """
    if method not in RETRACTIONS:
        raise ValueError(f"Unknown retraction: {method}")
    rank = A.shape[0]
    if method == "svd":
        new_A, new_B = truncated_svd(left, right, rank)
        return new_A, new_B, False

    # Cholesky tests the small factor Grams, not an m x n matrix.
    chol_A, info_A = torch.linalg.cholesky_ex(A @ A.T)
    chol_B, info_B = torch.linalg.cholesky_ex(B.T @ B)
    if info_A.item() or info_B.item():
        new_A, new_B = truncated_svd(left, right, rank)
        return new_A, new_B, True

    if method == "factor":
        # Canonical split of xi=Y-W. Cholesky solves avoid explicit inverses.
        xi_At = left @ (right @ A.T) - B @ (A @ A.T)
        dB = torch.cholesky_solve(xi_At.T, chol_A).T
        Bt_xi = (B.T @ left) @ right - (B.T @ B) @ A
        dA = torch.cholesky_solve(Bt_xi - (B.T @ dB) @ A, chol_B)
        return A + dA, B + dB, False

    U = torch.linalg.qr(B, mode="reduced").Q
    V = torch.linalg.qr(A.T, mode="reduced").Q
    YV = left @ (right @ V)
    if method == "orthographic":
        UtY = (U.T @ left) @ right
        core = U.T @ YV
        return torch.linalg.solve(core, UtY), YV, False

    # KSL projector splitting: K=Y V, U_new=qr(K), L=Y.T U_new.
    new_B = torch.linalg.qr(YV, mode="reduced").Q
    new_A = (new_B.T @ left) @ right
    return new_A, new_B, False


@torch.no_grad()
def retract_lora(
    A: torch.Tensor,
    B: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    method: str,
    gauge: str = "native",
) -> tuple:
    """Use the same xi=dB@A+B@dA for every map, excluding dB@dA."""
    if method not in RETRACTIONS or gauge not in ("native", "left_orthogonal"):
        raise ValueError(f"Invalid retraction/gauge: {method}/{gauge}")
    if method == "factor":
        new_A, new_B, fallback = A + dA, B + dB, False
    else:
        left = torch.cat([B, dB], dim=1)
        right = torch.cat([A + dA, A], dim=0)
        new_A, new_B, fallback = retract_candidate(A, B, left, right, method)
    if gauge == "left_orthogonal":
        new_A, new_B = left_orthogonalize(new_A, new_B)
    return new_A, new_B, fallback
