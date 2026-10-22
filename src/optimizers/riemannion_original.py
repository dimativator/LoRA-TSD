"""Authors' Riemannion update with selectable retractions.

Reference: Bogachevv/RiemanianFinetune, commit
1ca2e58de78c0e4d3a65acbe68ab7043dff53e6d.
Initialization is supplied by the existing local model pipeline.
"""

import torch

from .riemannian_lora import FixedRank, Riemannion


class RiemannionOriginal(Riemannion):
    def __init__(self, params, **defaults):
        # Reuse the local retraction dispatch, allowing every map with momentum.
        # The override below preserves the authors' previous-point references.
        defaults["transport_snapshot"] = True
        super().__init__(params, **defaults)
        self.transport_snapshot = False

    @torch.no_grad()
    def get_deltas(self, A, B, manifold: FixedRank) -> tuple:
        r = manifold.r
        dot_A = A.grad.T[:, r:]
        dot_B = B.grad.T[:r, :]
        point = [A.T[:, :r], B.T[:r, :]]
        vec = manifold.euclidian_to_reimanian(point, [dot_A, dot_B])
        return point, vec

    @torch.no_grad()
    def apply_momentum(self, point: list, vec: list, state: dict,
                       manifold: FixedRank, beta: float, nesterov: bool) -> list:
        if 'momentum' not in state:
            state['momentum'] = vec[0], vec[1]
            state['prev_point'] = point
            return vec

        m_vec = manifold.transport(state['prev_point'], point, state['momentum'])
        dot_A = beta * m_vec[0] + vec[0]
        dot_B = beta * m_vec[1] + vec[1]
        state['prev_point'] = point
        state['momentum'] = [dot_A, dot_B]

        # Upstream's Nesterov branch repeats its heavy-ball update.
        if nesterov:
            dot_A = beta * m_vec[0] + vec[0]
            dot_B = beta * m_vec[1] + vec[1]
        return dot_A, dot_B
