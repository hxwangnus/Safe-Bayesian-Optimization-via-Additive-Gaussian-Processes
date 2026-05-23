import math
import unittest

import gpytorch
import torch

from safectrlbo import SafeCtrlBO


def make_kernel():
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(0,)))
    kernel.initialize(outputscale=1.0)
    kernel.base_kernel.initialize(lengthscale=1.0)
    return kernel


def make_optimizer(**kwargs):
    init_X = torch.tensor([[0.0]], dtype=torch.double)
    init_Y_perf = torch.tensor([[1.0]], dtype=torch.double)
    init_Y_safe = kwargs.pop(
        "init_Y_safe",
        torch.tensor([[1.0, 1.0]], dtype=torch.double),
    )
    safety_threshold = kwargs.pop("safety_threshold", [0.0, 0.0])
    beta_fn = kwargs.pop("beta_fn", lambda _n: 0.0)
    return SafeCtrlBO(
        init_X=init_X,
        init_Y_perf=init_Y_perf,
        init_Y_safe=init_Y_safe,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        base_kernel=make_kernel(),
        safety_threshold=safety_threshold,
        beta_fn=beta_fn,
        device="cpu",
        likelihood_noise=1e-4,
        **kwargs,
    )


class SafeCtrlBOTests(unittest.TestCase):
    def test_safe_set_requires_all_safety_constraints(self):
        algo = make_optimizer()
        X_cand = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.double)
        safety_means = {
            id(algo.models_g[0]): torch.tensor([0.2, 0.1, -0.1, 0.3], dtype=torch.double),
            id(algo.models_g[1]): torch.tensor([0.1, -0.2, 0.2, 0.05], dtype=torch.double),
        }

        def fake_posterior(model, _likelihood, Xtest):
            if id(model) == id(algo.model_f):
                return torch.zeros(Xtest.shape[0], dtype=torch.double), torch.ones(Xtest.shape[0], dtype=torch.double)
            return safety_means[id(model)], torch.zeros(Xtest.shape[0], dtype=torch.double)

        algo.posterior_mean_std = fake_posterior

        sets = algo._get_sets(X_cand, beta=0.0)

        self.assertEqual(sets["safe_mask"].tolist(), [True, False, False, True])
        self.assertTrue(torch.equal(sets["S"], X_cand[[0, 3]]))

    def test_empty_relaxed_boundary_falls_back_to_smallest_safe_margin(self):
        algo = make_optimizer(tau=0.01)
        X_cand = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double)
        safety_means = {
            id(algo.models_g[0]): torch.tensor([0.5, 0.2, -0.1], dtype=torch.double),
            id(algo.models_g[1]): torch.tensor([0.4, 0.3, 0.2], dtype=torch.double),
        }

        def fake_posterior(model, _likelihood, Xtest):
            if id(model) == id(algo.model_f):
                return torch.zeros(Xtest.shape[0], dtype=torch.double), torch.ones(Xtest.shape[0], dtype=torch.double)
            return safety_means[id(model)], torch.zeros(Xtest.shape[0], dtype=torch.double)

        algo.posterior_mean_std = fake_posterior

        sets = algo._get_sets(X_cand, beta=0.0)

        self.assertEqual(sets["safe_mask"].tolist(), [True, True, False])
        self.assertEqual(sets["boundary_mask"].tolist(), [False, True, False])
        self.assertTrue(torch.equal(sets["B"], X_cand[[1]]))

    def test_default_beta_uses_paper_confidence_width_form(self):
        algo = make_optimizer(
            init_Y_safe=torch.tensor([[1.0]], dtype=torch.double),
            safety_threshold=0.0,
            beta_fn=None,
            rkhs_bound=1.5,
            noise_bound=0.2,
            delta=0.1,
            information_gain_fn=lambda _t: 2.0,
        )

        expected = 1.5 + 0.2 * math.sqrt(2.0 * (2.0 + 1.0 + math.log(10.0)))

        self.assertAlmostEqual(float(algo.beta_fn(4)), expected, places=12)
        self.assertAlmostEqual(float(algo._beta_width(algo.beta_fn(4), torch.double)), expected, places=12)

    def test_expansion_can_score_boundary_by_safety_uncertainty(self):
        algo = make_optimizer(expansion_uncertainty="safety", switch_time=10)
        X_static = torch.tensor([[0.25], [0.75]], dtype=torch.double)
        sets = {
            "S": X_static,
            "B": X_static,
            "safe_mask": torch.tensor([True, True]),
            "boundary_mask": torch.tensor([True, True]),
            "u_f": torch.tensor([0.0, 0.0], dtype=torch.double),
            "sigma_f": torch.tensor([10.0, 1.0], dtype=torch.double),
            "l_g": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.double),
            "sigma_g": torch.tensor([[1.0, 2.0], [5.0, 0.1]], dtype=torch.double),
            "safety_margin": torch.tensor([1.0, 1.0], dtype=torch.double),
        }
        algo._get_sets = lambda _X, _beta: sets

        x_next, mode, _sets = algo.suggest(num_candidates=2)

        self.assertEqual(mode, "expansion")
        self.assertTrue(torch.equal(x_next, X_static[[1]]))

    def test_observe_appends_all_safety_constraints(self):
        algo = make_optimizer()

        algo.observe(
            x_new=torch.tensor([[0.5]], dtype=torch.double),
            y_perf_new=torch.tensor([[2.0]], dtype=torch.double),
            y_safe_new=torch.tensor([[0.5, 0.6]], dtype=torch.double),
        )

        self.assertEqual(algo.n_iter, 2)
        self.assertEqual(tuple(algo.Yg.shape), (2, 2))
        self.assertTrue(torch.equal(algo.Yg[-1], torch.tensor([0.5, 0.6], dtype=torch.double)))
        self.assertEqual(algo.models_g[0].train_targets.numel(), 2)
        self.assertEqual(algo.models_g[1].train_targets.numel(), 2)


if __name__ == "__main__":
    unittest.main()
