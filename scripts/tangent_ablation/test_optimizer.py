"""Numerical checks for direction isolation and the reported update spectra."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from optimizer import (TangentAblation, VARIANTS, factored_singular_values,
                       rectangular_sign)
from tsd.lora_tsd import _ns_sq, tangent_projected_grad
from tsd.lora_tsd_efficient import LoRATSDEfficient


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.A = torch.randn(3, 13)
        self.B = torch.randn(11, 3)
        self.G = torch.randn(11, 13)

    def make_optimizer(self, variant, path=None, **kwargs):
        A = torch.nn.Parameter(self.A.clone())
        B = torch.nn.Parameter(self.B.clone())
        A.grad = self.B.t() @ self.G
        B.grad = self.G @ self.A.t()
        optimizer = TangentAblation([{"params": [A, B], "name": "layers.0.q_proj"}],
                                   variant, lr=.003, spectrum_path=path,
                                   use_momentum=False, balance=False,
                                   max_delta_norm=100., **kwargs)
        return optimizer, A, B

    def dense_direction(self, variant):
        A, B, G = self.A, self.B, self.G
        Q = torch.linalg.qr(B, mode="reduced")[0]
        V = torch.linalg.qr(A.t(), mode="reduced")[0]
        PB, PA = Q @ Q.t(), V @ V.t()

        def project(matrix):
            return PB @ matrix + matrix @ PA - PB @ matrix @ PA

        if variant == "factorwise":
            return B @ rectangular_sign(B.t() @ G) + rectangular_sign(G @ A.t()) @ A
        direction = project(G)
        for _ in range({"frobenius": 0, "one_step": 1, "coupled": 5}[variant]):
            # Same polynomial in dense coordinates gives an independent oracle.
            direction = project(rectangular_sign(direction))
        return direction

    def test_tangent_projection_matches_dense_projector(self):
        Q = torch.linalg.qr(self.B, mode="reduced")[0]
        V = torch.linalg.qr(self.A.t(), mode="reduced")[0]
        PB, PA = Q @ Q.t(), V @ V.t()
        actual = tangent_projected_grad(self.A, self.B,
                                        self.B.t() @ self.G, self.G @ self.A.t())
        expected = PB @ self.G + self.G @ PA - PB @ self.G @ PA
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_rectangular_polynomial_matches_square_implementation(self):
        matrix = torch.randn(7, 7)
        torch.testing.assert_close(rectangular_sign(matrix), _ns_sq(matrix), atol=1e-6, rtol=1e-6)
        for shape in [(5, 11), (11, 5)]:
            matrix = torch.randn(*shape)
            torch.testing.assert_close(rectangular_sign(matrix.t()).t(),
                                       rectangular_sign(matrix), atol=2e-6, rtol=2e-6)

    def test_small_core_spectrum_matches_dense_svd(self):
        left, right = torch.randn(11, 6), torch.randn(6, 13)
        actual = factored_singular_values(left, right)
        expected = torch.linalg.svdvals(left @ right)[:6]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_all_variants_share_canonical_reconstruction(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                optimizer, A, B = self.make_optimizer(variant)
                optimizer.step()
                dA, dB = A.detach() - self.A, B.detach() - self.B
                actual = self.B @ dA + dB @ self.A
                expected = -.003 * self.dense_direction(variant)
                torch.testing.assert_close(actual, expected, atol=6e-6, rtol=2e-4)
                # The canonical split allocates the row(A) component to dB A.
                PA = self.A.t() @ torch.linalg.solve(self.A @ self.A.t(), self.A)
                torch.testing.assert_close(dB @ self.A, expected @ PA, atol=6e-6, rtol=2e-4)

    def test_reported_spectra_include_quadratic_retraction_term(self):
        with tempfile.TemporaryDirectory() as directory:
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    path = Path(directory) / (variant + ".jsonl")
                    optimizer, A, B = self.make_optimizer(variant, path)
                    optimizer.step()
                    dA, dB = A.detach() - self.A, B.detach() - self.B
                    tangent = self.B @ dA + dB @ self.A
                    applied = B.detach() @ A.detach() - self.B @ self.A
                    torch.testing.assert_close(applied, tangent + dB @ dA, atol=2e-6, rtol=1e-4)
                    row = json.loads(path.read_text())
                    for key, dense in [("tangent_update", tangent), ("applied_update", applied)]:
                        expected = torch.linalg.svdvals(dense)[:6]
                        torch.testing.assert_close(torch.tensor(row[key]["singular_values"]), expected,
                                                   atol=4e-6, rtol=2e-4)

    def test_spectrum_with_balancing_and_momentum_is_actual_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrum.jsonl"
            A, B = torch.nn.Parameter(self.A.clone()), torch.nn.Parameter(self.B.clone())
            optimizer = TangentAblation([{"params": [A, B]}], "coupled", spectrum_path=path,
                                       spectrum_steps=[1, 2])
            for step in [1, 2]:
                old_A, old_B = A.detach().clone(), B.detach().clone()
                A.grad, B.grad = B.detach().t() @ self.G, self.G @ A.detach().t()
                optimizer.step()
                row = json.loads(path.read_text().splitlines()[step - 1])
                expected = torch.linalg.svdvals(B.detach() @ A.detach() - old_B @ old_A)[:6]
                torch.testing.assert_close(torch.tensor(row["applied_update"]["singular_values"]),
                                           expected, atol=4e-6, rtol=2e-4)

    def test_coupled_matches_existing_optimizer_with_momentum_and_balancing(self):
        pairs = [(torch.nn.Parameter(self.A.clone()), torch.nn.Parameter(self.B.clone()))
                 for _ in range(2)]
        control = LoRATSDEfficient([{"params": list(pairs[0])}], lr_A=.005, lr_B=.005,
                                  momentum_cf=.95, ns_steps=5, ball_iters=5, balance=True,
                                  max_delta_norm=.1, log_optimizer_stats=False)
        ablation = TangentAblation([{"params": list(pairs[1])}], "coupled")
        for _ in range(3):
            for optimizer, (A, B) in zip([control, ablation], pairs):
                A.grad, B.grad = B.detach().t() @ self.G, self.G @ A.detach().t()
                optimizer.step()
            for actual, expected in zip(pairs[1], pairs[0]):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_zero_gradients_are_finite_and_do_not_move_product(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                optimizer, A, B = self.make_optimizer(variant)
                A.grad.zero_()
                B.grad.zero_()
                optimizer.step()
                self.assertTrue(torch.isfinite(A).all() and torch.isfinite(B).all())
                torch.testing.assert_close(B @ A, self.B @ self.A, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
