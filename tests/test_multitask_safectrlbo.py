import unittest
import math

import gpytorch
import torch

from multitask_safectrlbo import MultiTaskSafeCtrlBO


def make_kernel():
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
    kernel.initialize(outputscale=1.0)
    kernel.base_kernel.initialize(lengthscale=torch.tensor([1.0, 1.0], dtype=torch.double))
    return kernel


def make_kernel_components():
    k0 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(0,), ard_num_dims=1))
    k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(1,), ard_num_dims=1))
    for kernel in (k0, k1):
        kernel.initialize(outputscale=0.5)
        kernel.base_kernel.initialize(lengthscale=torch.ones(1, 1, dtype=torch.double))
    return [k0, k1]


def make_optimizer(num_initial=1, beta_fn=lambda _n: 0.0, **kwargs):
    base_kernel = kwargs.pop("base_kernel", make_kernel())
    init_X = torch.tensor(
        [[0.4, 0.4], [0.45, 0.42], [0.35, 0.38]][:num_initial],
        dtype=torch.double,
    )
    init_Y_perf = torch.tensor([[0.4, 0.3, 0.2]], dtype=torch.double).repeat(num_initial, 1)
    init_Y_safe = torch.tensor(
        [[
            [0.5, 0.4],
            [0.4, 0.3],
            [0.3, 0.2],
        ]],
        dtype=torch.double,
    ).repeat(num_initial, 1, 1)
    optimizer_kwargs = dict(
        init_X=init_X,
        init_Y_perf=init_Y_perf,
        init_Y_safe=init_Y_safe,
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        base_kernel=base_kernel,
        safety_threshold=torch.zeros((3, 2), dtype=torch.double),
        mode_names=("free", "transition", "contact"),
        mode_weights=torch.tensor([0.2, 0.3, 0.5], dtype=torch.double),
        device="cpu",
        likelihood_noise=1e-4,
        **kwargs,
    )
    if beta_fn is not None:
        optimizer_kwargs["beta_fn"] = beta_fn
    return MultiTaskSafeCtrlBO(**optimizer_kwargs)


class MultiTaskSafeCtrlBOTests(unittest.TestCase):
    def test_default_surrogate_is_lmc(self):
        algo = make_optimizer()

        self.assertEqual(algo.multitask_kernel, "lmc")

    def test_safe_set_intersects_all_modes_and_constraints(self):
        algo = make_optimizer()
        X_cand = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.double)

        def fake_perf(_X):
            q = _X.shape[0]
            return (
                torch.zeros((q, 3), dtype=torch.double),
                torch.ones((q, 3), dtype=torch.double),
            )

        safety_mean = torch.tensor(
            [
                [[0.2, 0.1], [0.1, 0.2], [0.1, 0.1]],
                [[0.2, 0.1], [-0.1, 0.2], [0.1, 0.1]],
                [[0.2, 0.1], [0.1, 0.2], [0.1, -0.2]],
            ],
            dtype=torch.double,
        )

        def fake_safety(_X):
            return safety_mean, torch.zeros_like(safety_mean)

        algo.posterior_perf_by_mode = fake_perf
        algo.posterior_safety_by_mode = fake_safety

        sets = algo._get_sets(X_cand, beta=0.0)

        self.assertEqual(sets["safe_mask"].tolist(), [True, False, False])
        self.assertTrue(torch.equal(sets["S"], X_cand[[0]]))

    def test_boundary_fallback_uses_smallest_mode_safety_margin(self):
        algo = make_optimizer(tau=0.01)
        X_cand = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.double)

        algo.posterior_perf_by_mode = lambda _X: (
            torch.zeros((_X.shape[0], 3), dtype=torch.double),
            torch.ones((_X.shape[0], 3), dtype=torch.double),
        )
        safety_mean = torch.tensor(
            [
                [[0.5, 0.4], [0.4, 0.3], [0.3, 0.2]],
                [[0.3, 0.2], [0.2, 0.1], [0.2, 0.1]],
                [[-0.1, 0.2], [0.2, 0.3], [0.3, 0.2]],
            ],
            dtype=torch.double,
        )
        algo.posterior_safety_by_mode = lambda _X: (safety_mean, torch.zeros_like(safety_mean))

        sets = algo._get_sets(X_cand, beta=0.0)

        self.assertEqual(sets["safe_mask"].tolist(), [True, True, False])
        self.assertEqual(sets["boundary_mask"].tolist(), [False, True, False])
        self.assertTrue(torch.equal(sets["B"], X_cand[[1]]))

    def test_optimization_uses_weighted_mode_utility(self):
        algo = make_optimizer(switch_time=0)
        X_static = torch.tensor([[0.25, 0.25], [0.75, 0.75]], dtype=torch.double)
        sets = {
            "S": X_static,
            "B": X_static,
            "safe_mask": torch.tensor([True, True]),
            "boundary_mask": torch.tensor([True, True]),
            "utility_u": torch.tensor([0.1, 0.9], dtype=torch.double),
            "utility_mean": torch.tensor([0.1, 0.9], dtype=torch.double),
            "utility_std": torch.tensor([0.0, 0.0], dtype=torch.double),
            "sigma_g": torch.ones((2, 3, 2), dtype=torch.double),
            "l_g": torch.ones((2, 3, 2), dtype=torch.double),
            "safety_margin": torch.ones(2, dtype=torch.double),
        }
        algo._get_sets = lambda *_args, **_kwargs: sets

        x_next, mode, _sets = algo.suggest(num_candidates=2)

        self.assertEqual(mode, "optimization")
        self.assertTrue(torch.equal(x_next, X_static[[1]]))

    def test_switch_time_counts_bo_steps_not_initial_observations(self):
        algo = make_optimizer(num_initial=3, switch_time=2)
        X_static = torch.tensor([[0.25, 0.25], [0.75, 0.75]], dtype=torch.double)
        sets = {
            "S": X_static,
            "B": X_static,
            "safe_mask": torch.tensor([True, True]),
            "boundary_mask": torch.tensor([True, True]),
            "utility_u": torch.tensor([0.1, 0.9], dtype=torch.double),
            "utility_mean": torch.tensor([0.1, 0.9], dtype=torch.double),
            "utility_std": torch.tensor([0.0, 1.0], dtype=torch.double),
            "sigma_g": torch.ones((2, 3, 2), dtype=torch.double),
            "l_g": torch.ones((2, 3, 2), dtype=torch.double),
            "safety_margin": torch.ones(2, dtype=torch.double),
        }
        algo._get_sets = lambda *_args, **_kwargs: sets

        _x_first, mode_first, _sets_first = algo.suggest(num_candidates=2)
        self.assertEqual(mode_first, "expansion")

        algo.observe(
            x_new=torch.tensor([[0.8, 0.8]], dtype=torch.double),
            y_perf_new=torch.tensor([[0.8, 0.7, 0.6]], dtype=torch.double),
            y_safe_new=torch.tensor([[[0.4, 0.3], [0.35, 0.25], [0.3, 0.2]]], dtype=torch.double),
        )
        _x_second, mode_second, _sets_second = algo.suggest(num_candidates=2)
        self.assertEqual(mode_second, "expansion")

        algo.observe(
            x_new=torch.tensor([[0.7, 0.7]], dtype=torch.double),
            y_perf_new=torch.tensor([[0.7, 0.6, 0.5]], dtype=torch.double),
            y_safe_new=torch.tensor([[[0.3, 0.2], [0.25, 0.2], [0.2, 0.15]]], dtype=torch.double),
        )
        _x_third, mode_third, _sets_third = algo.suggest(num_candidates=2)
        self.assertEqual(mode_third, "optimization")

    def test_utility_std_uses_conservative_weighted_sum(self):
        algo = make_optimizer()
        mean_f = torch.zeros((1, 3), dtype=torch.double)
        std_f = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.double)

        utility_mean, utility_std = algo._utility_mean_std(mean_f, std_f)

        self.assertTrue(torch.allclose(utility_mean, torch.zeros(1, dtype=torch.double)))
        self.assertTrue(torch.allclose(utility_std, torch.tensor([2.8], dtype=torch.double)))

    def test_default_safety_beta_uses_mode_constraint_union_correction(self):
        algo = make_optimizer(
            beta_fn=None,
            rkhs_bound=0.1,
            noise_bound=0.2,
            delta=0.05,
            information_gain_fn=lambda _t: torch.tensor(0.0, dtype=torch.double),
        )

        beta_f = float(algo.beta_f_fn(4))
        beta_g = float(algo.beta_g_fn(4))

        expected_f = 0.1 + 0.2 * math.sqrt(2.0 * (1.0 + math.log(1.0 / 0.05)))
        expected_g = 0.1 + 0.2 * math.sqrt(2.0 * (1.0 + math.log(6.0 / 0.05)))
        self.assertAlmostEqual(beta_f, expected_f, places=12)
        self.assertAlmostEqual(beta_g, expected_g, places=12)
        self.assertGreater(beta_g, beta_f)

    def test_observe_updates_multitask_training_shapes(self):
        algo = make_optimizer()

        algo.observe(
            x_new=torch.tensor([[0.8, 0.8]], dtype=torch.double),
            y_perf_new=torch.tensor([[0.8, 0.7, 0.6]], dtype=torch.double),
            y_safe_new=torch.tensor(
                [[[0.4, 0.3], [0.35, 0.25], [0.3, 0.2]]],
                dtype=torch.double,
            ),
        )

        self.assertEqual(algo.n_iter, 2)
        self.assertEqual(tuple(algo.Yf.shape), (2, 3))
        self.assertEqual(tuple(algo.Yg.shape), (2, 3, 2))
        self.assertEqual(tuple(algo.model_f.train_targets.shape), (2, 3))
        self.assertEqual(tuple(algo.models_g[0].train_targets.shape), (2, 3))
        self.assertEqual(tuple(algo.models_g[1].train_targets.shape), (2, 3))

    def test_lmc_surrogate_initializes_all_mode_models(self):
        algo = make_optimizer(
            base_kernel=make_kernel_components(),
            multitask_kernel="lmc",
            task_rank=1,
        )

        self.assertEqual(algo.multitask_kernel, "lmc")
        self.assertEqual(len(algo.model_f.covar_module.kernels), 2)
        self.assertEqual(len(algo.models_g[0].covar_module.kernels), 2)

    def test_observe_partial_fills_missing_modes_conservatively(self):
        algo = make_optimizer()

        algo.observe_partial(
            x_new=torch.tensor([[0.8, 0.8]], dtype=torch.double),
            y_perf_new=torch.tensor([[0.7, 0.1]], dtype=torch.double),
            y_safe_new=torch.tensor([[[0.2, 0.1], [-0.3, -0.4]]], dtype=torch.double),
            observed_modes=("free", "transition"),
            missing_perf_value=-2.0,
            missing_safety_value=-0.75,
        )

        self.assertEqual(algo.n_iter, 2)
        self.assertEqual(algo.bo_steps, 1)
        self.assertTrue(torch.allclose(algo.Yf[-1], torch.tensor([0.7, 0.1, -2.0], dtype=torch.double)))
        self.assertTrue(
            torch.allclose(
                algo.Yg[-1],
                torch.tensor(
                    [[0.2, 0.1], [-0.3, -0.4], [-0.75, -0.75]],
                    dtype=torch.double,
                ),
            )
        )

    def test_observe_partial_rejects_non_abort_missing_modes(self):
        algo = make_optimizer()

        with self.assertRaises(NotImplementedError):
            algo.observe_partial(
                x_new=torch.tensor([[0.8, 0.8]], dtype=torch.double),
                y_perf_new=torch.tensor([[0.7, 0.1]], dtype=torch.double),
                y_safe_new=torch.tensor([[[0.2, 0.1], [0.3, 0.2]]], dtype=torch.double),
                observed_modes=("free", "transition"),
                missing_reason="no_contact",
            )


if __name__ == "__main__":
    unittest.main()
