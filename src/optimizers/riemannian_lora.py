import math

import torch
from torch.optim import Optimizer


class FixedRank:
    """Fixed-rank manifold operations used by the Riemannian baselines."""

    def __init__(self, r: int, svd_driver: str = None):
        self.svd_driver = svd_driver
        self.r = r

    def norm2(self, point, vec):
        dot_A, dot_B = vec
        _, B = point
        return [torch.norm(dot_A)**2, torch.norm(dot_B)**2]

    def retraction(self, point, tangent_vec):
        A_part, B_part = self.embedding(point, tangent_vec, ambient=False)
        Q_l, R_l = torch.linalg.qr(A_part)
        Q_r, R_r = torch.linalg.qr(B_part.T)
        T = R_l @ R_r.T

        U, S, Vt = torch.linalg.svd(T, driver=self.svd_driver)
        U = U[:, :self.r]
        S = S[:self.r]
        Vt = Vt[:self.r, :]

        new_A = Q_l @ U
        new_B = torch.diag(S) @ Vt @ Q_r.T
        return [new_A, new_B]

    def embedding(self, point: list, vec: list, ambient: bool = True):
        U_l, _ = point
        _, Vt_r = self.to_right(point)

        if ambient:
            return U_l @ vec[1] + vec[0] @ Vt_r

        left  = torch.cat([U_l, vec[0]],  dim=1)
        right = torch.cat([vec[1], Vt_r], dim=0)
        return [left, right]

    def to_right(self, point: list):
        A, B = point
        Q, R = torch.linalg.qr(B.T)
        B = Q.T
        A = A @ R.T

        return [A, B]

    def euclidian_to_reimanian(self, point: list, vec: list) -> list:
        A, B = point
        dot_A, dot_B = vec

        dot_A -= A @ (A.T @ dot_A)

        return dot_A, dot_B

    def transport(self, point_X: list, point_Y: list, vec_X: list):
        U_l, _ = point_Y
        _, Vt_r = self.to_right(point_Y)

        A_part, B_part = self.embedding(point_X, vec_X, ambient=False)
        dot_U = (A_part - U_l @ (U_l.T @ A_part)) @ (B_part @ Vt_r.T)
        dot_V = (U_l.T @ A_part) @ B_part
        return [dot_U, dot_V]


class RiemannianLora(Optimizer):
    def __init__(self, params, **defaults):

        lr = defaults.get('lr', 0.001)
        betas = defaults.get('betas', [0.9, 0.999])
        super().__init__(params, {'lr': lr, 'betas': betas})
        self.last_logs = None

    @torch.no_grad()
    def apply_second_momentum(self, point: list, grad_vec: list, momentum_vec: list,
                               state: dict, beta: list, manifold: FixedRank):
        norm = manifold.norm2(point=point, vec=grad_vec)

        if 'second_momentum' not in state:
            state['second_momentum'] = [norm[0] * (1 - beta[1]), norm[1] * (1 - beta[1])]

            addm = [1 / (torch.sqrt(state['second_momentum'][0]) + 1e-8),
                    1 / (torch.sqrt(state['second_momentum'][1]) + 1e-8)]
            dot_A = momentum_vec[0] * addm[0]
            dot_B = momentum_vec[1] * addm[1]

            return dot_A, dot_B

        state['second_momentum'] = [
            beta[1] * state['second_momentum'][0] + (1 - beta[1]) * norm[0],
            beta[1] * state['second_momentum'][1] + (1 - beta[1]) * norm[1],
        ]

        addm = [1 / (torch.sqrt(state['second_momentum'][0]) + 1e-8),
                1 / (torch.sqrt(state['second_momentum'][1]) + 1e-8)]
        dot_A = momentum_vec[0] * addm[0]
        dot_B = momentum_vec[1] * addm[1]

        return dot_A, dot_B

    @torch.no_grad()
    def get_deltas(self, A, B, manifold: FixedRank) -> tuple:
        dot_A = A.grad.T
        dot_B = B.grad.T

        r = manifold.r
        dot_A = dot_A[:, :r]
        dot_B = dot_B[:r, :]

        A_eff = A.T[:, :r]
        B_eff = B.T[:r, :]

        point = [A_eff, B_eff]

        vec = manifold.euclidian_to_reimanian(
            point=point,
            vec=[dot_A, dot_B]
        )

        return point, vec

    @torch.no_grad()
    def apply_momentum(self, point: list, vec: list, state: dict,
                       manifold: FixedRank, beta: float) -> list:

        if 'momentum' not in state:
            state['momentum'] = (vec[0] * (1 - beta), vec[1] * (1 - beta))
            state['prev_point'] = point
            return vec

        m_vec = manifold.transport(
            point_X=state['prev_point'],
            point_Y=point,
            vec_X=state['momentum'],
        )
        dot_A = beta * m_vec[0] + (1 - beta) * vec[0]
        dot_B = beta * m_vec[1] + (1 - beta) * vec[1]

        state['prev_point'] = point
        state['momentum'] = [dot_A, dot_B]

        return dot_A, dot_B

    @torch.no_grad()
    def step(self, closure=None):
        step_logs = [] if any(
            group.get("log_optimizer_stats", True)
            for group in self.param_groups
        ) else None

        for group_idx, group in enumerate(self.param_groups):
            A_idx = group['param_names'].index('lora_A.weight')
            B_idx = group['param_names'].index('lora_B.weight')

            A = group['params'][A_idx]
            B = group['params'][B_idx]
            r = A.shape[0] // 2

            if A.grad is None or B.grad is None:
                continue

            lr   = group['lr']
            beta = group['betas']

            state = self.state[A]
            if len(state) == 0:
                state['step'] = 0
                state['manifold'] = FixedRank(r=r)

            manifold = state['manifold']
            point, grad_vec = self.get_deltas(A, B, manifold=manifold)

            momentum_vec = self.apply_momentum(
                point=point,
                vec=grad_vec,
                state=state,
                manifold=manifold,
                beta=beta[0],
            )

            dot_A, dot_B = self.apply_second_momentum(
                point=point,
                grad_vec=grad_vec,
                momentum_vec=momentum_vec,
                state=state,
                beta=beta,
                manifold=manifold,
            )

            new_A, new_B = manifold.retraction(
                point=point,
                tangent_vec=[-lr * dot_A, point[1] - lr * dot_B],
            )

            A_zero = torch.zeros_like(new_A)
            B_orth = torch.linalg.qr(new_B.T).Q.T

            new_A = torch.cat([new_A, A_zero], dim=1)
            new_B = torch.cat([new_B, B_orth], dim=0)

            group['params'][A_idx].copy_(new_A.T)
            group['params'][B_idx].copy_(new_B.T)

            if step_logs is not None:
                step_logs.append({
                    "group_idx":  group_idx,
                    "A_F_norm": point[0].norm().item(),
                    "B_F_norm": point[1].norm().item(),
                    "G_A_F_norm": grad_vec[0].norm().item(),
                    "G_B_F_norm": grad_vec[1].norm().item(),
                    "dA_F_norm": dot_A.norm().item(),
                    "dB_F_norm": dot_B.norm().item(),
                })

            state['step'] += 1

        self.last_logs = step_logs


class RiemannianSGD(RiemannianLora):
    def __init__(self, params, **defaults):

        lr       = defaults.get('lr', 0.001)
        momentum = defaults.get('momentum', 0.0)

        betas = [momentum, 0.0]
        super().__init__(params, **{'lr': lr, 'betas': betas})

    @torch.no_grad()
    def apply_second_momentum(self, point: list, grad_vec: list, momentum_vec: list,
                               state: dict, beta: list, manifold: FixedRank):
        return momentum_vec


class Riemannion(Optimizer):
    def __init__(self, params, **defaults):
        lr           = defaults.get('lr', 0.001)
        momentum     = defaults.get('momentum', 0.0)
        nesterov     = defaults.get('nesterov', False)
        nuclear_scale = defaults.get('nuclear_scale', False)
        weight_decay = defaults.get('weight_decay', 0.0)
        super().__init__(params, {
            'lr': lr, 'momentum': momentum, 'nesterov': nesterov,
            'nuclear_scale': nuclear_scale, 'weight_decay': weight_decay,
        })
        self.svd_driver = None

    @torch.no_grad()
    def get_deltas(self, A, B, manifold: FixedRank) -> tuple:
        # Riemannion updates the left factor through the retraction geometry.
        dot_A = torch.zeros_like(A.T[:, :manifold.r])
        dot_B = B.grad.T[:manifold.r, :]

        r = manifold.r
        A_eff = A.T[:, :r]
        B_eff = B.T[:r, :]
        point = [A_eff, B_eff]

        vec = manifold.euclidian_to_reimanian(point=point, vec=[dot_A, dot_B])
        return point, vec

    def adjust_lr_for_muon(self, lr, m_shape: int, n_shape: int, r_shape: int,
                           nuclear_scale: bool = False):
        if nuclear_scale:
            return lr
        unifying_constant = 0.2
        adjusted_ratio = unifying_constant * math.sqrt(m_shape * n_shape / r_shape)
        return lr * adjusted_ratio

    @torch.no_grad()
    def apply_momentum(self, point: list, vec: list, state: dict,
                       manifold: FixedRank, beta: float, nesterov: bool) -> list:
        if 'momentum' not in state:
            state['momentum'] = (vec[0], vec[1])
            state['prev_point'] = point
            return vec

        m_vec = manifold.transport(
            point_X=state['prev_point'],
            point_Y=point,
            vec_X=state['momentum'],
        )
        dot_A = beta * m_vec[0] + vec[0]
        dot_B = beta * m_vec[1] + vec[1]

        state['prev_point'] = point
        state['momentum'] = [dot_A, dot_B]

        if nesterov:
            dot_A = beta * m_vec[0] + (1 + beta) * vec[0]
            dot_B = beta * m_vec[1] + (1 + beta) * vec[1]

        return dot_A, dot_B

    def apply_muon(self, point: list, momentum: list, manifold: FixedRank,
                   nuclear_scale: bool = False) -> list:
        left, right = manifold.embedding(point, momentum, ambient=False)

        Q_l, R_l = torch.linalg.qr(left)
        Q_r, R_r = torch.linalg.qr(right.T)
        T = R_l @ R_r.T
        U, S, Vt = torch.linalg.svd(T, driver=self.svd_driver)

        A_part = Q_l @ U
        B_part = Vt @ Q_r.T

        U_l, _ = point
        _, Vt_r = manifold.to_right(point)

        dot_U = (A_part - U_l @ (U_l.T @ A_part)) @ (B_part @ Vt_r.T)
        dot_V = (U_l.T @ A_part) @ B_part

        if nuclear_scale:
            scale = S.sum()
            dot_U.mul_(scale)
            dot_V.mul_(scale)

        return [dot_U, dot_V]

    def apply_weight_decay(self, point: list, vec: list, weight_decay: float) -> list:
        return [
            vec[0],
            vec[1] + weight_decay * point[1],
        ]

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            A_idx = group['param_names'].index('lora_A.weight')
            B_idx = group['param_names'].index('lora_B.weight')

            A = group['params'][A_idx]
            B = group['params'][B_idx]
            r = A.shape[0] // 2

            if A.grad is None or B.grad is None:
                continue

            lr           = group['lr']
            beta         = group['momentum']
            nesterov     = group['nesterov']
            nuclear_scale = group['nuclear_scale']
            weight_decay = group['weight_decay']

            lr = self.adjust_lr_for_muon(
                lr, m_shape=B.shape[0], n_shape=A.shape[1], r_shape=r,
                nuclear_scale=nuclear_scale,
            )

            state = self.state[A]
            if len(state) == 0:
                state['step'] = 0
                state['manifold'] = FixedRank(r=r, svd_driver=self.svd_driver)

            manifold = state['manifold']
            point, grad_vec = self.get_deltas(A, B, manifold=manifold)

            momentum_vec = self.apply_momentum(
                point=point, vec=grad_vec,
                state=state, manifold=manifold,
                beta=beta, nesterov=nesterov,
            )

            dot_A, dot_B = self.apply_muon(
                point=point, momentum=momentum_vec,
                manifold=manifold, nuclear_scale=nuclear_scale,
            )

            dot_A, dot_B = self.apply_weight_decay(
                point=point, vec=[dot_A, dot_B], weight_decay=weight_decay,
            )

            new_A, new_B = manifold.retraction(
                point=point,
                tangent_vec=[-lr * dot_A, point[1] - lr * dot_B],
            )

            A_zero = torch.zeros_like(new_A)
            B_orth = torch.linalg.qr(new_B.T).Q.T

            new_A = torch.cat([new_A, A_zero], dim=1)
            new_B = torch.cat([new_B, B_orth], dim=0)

            group['params'][A_idx].copy_(new_A.T)
            group['params'][B_idx].copy_(new_B.T)
            state['step'] += 1
