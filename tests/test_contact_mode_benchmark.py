import unittest
from unittest.mock import patch
import inspect

import numpy as np
import gpytorch
import torch

from contact_mode_benchmark import (
    INPUT_DIM,
    contact_rich_12d_torch,
    fused_single_task_observations,
    is_certified_suggestion_mode,
    make_group_additive_kernel,
    make_group_kernel_components,
    make_surrogate_kernel,
    run_experiment,
    sample_initial_safe_points,
)


class ContactModeBenchmarkTests(unittest.TestCase):
    def test_run_experiment_default_surrogate_is_lmc(self):
        self.assertEqual(inspect.signature(run_experiment).parameters["surrogate"].default, "lmc")

    def test_group_kernel_components_match_additive_kernel_structure(self):
        components = make_group_kernel_components(torch.device("cpu"), torch.double)
        additive = make_group_additive_kernel(torch.device("cpu"), torch.double)

        self.assertEqual(len(components), 5)
        self.assertEqual(len(additive.kernels), 5)

    def test_make_surrogate_kernel_selects_icm_or_lmc_inputs(self):
        icm_kernel = make_surrogate_kernel("icm", torch.device("cpu"), torch.double)
        lmc_kernels = make_surrogate_kernel("lmc", torch.device("cpu"), torch.double)

        self.assertIsInstance(icm_kernel, gpytorch.kernels.AdditiveKernel)
        self.assertEqual(len(lmc_kernels), 5)

    def test_initial_safe_sampling_uses_noise_free_safety_decision(self):
        def fake_contact_function(x, noise_std=0.0, **_kwargs):
            perf = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)
            safe_value = 1.0 if noise_std > 0.0 else -1.0
            safe = torch.full((x.shape[0], 3, 2), safe_value, dtype=x.dtype, device=x.device)
            if torch.allclose(x[0], torch.tensor([0.25, 0.25, 0.30, 0.30, 0.25, 0.25, 0.70, 0.70, 0.30, 0.30, 0.35, 0.35], dtype=x.dtype, device=x.device)):
                safe.fill_(1.0)
            return perf, safe

        with patch("contact_mode_benchmark.contact_rich_12d_torch", side_effect=fake_contact_function):
            with self.assertRaises(RuntimeError):
                sample_initial_safe_points(
                    num_initial=2,
                    run_rng=np.random.default_rng(0),
                    device=torch.device("cpu"),
                    dtype=torch.double,
                    noise_std=1.0,
                    max_attempts=1,
                )

    def test_certified_suggestion_mode_excludes_empirical_fallbacks(self):
        self.assertTrue(is_certified_suggestion_mode("expansion"))
        self.assertTrue(is_certified_suggestion_mode("optimization_local_retry"))
        self.assertTrue(is_certified_suggestion_mode("safe_fallback"))
        self.assertFalse(is_certified_suggestion_mode("empirical_expansion_local_retry"))
        self.assertFalse(is_certified_suggestion_mode("empirical_safe_fallback"))

    def test_fused_single_task_observations_use_weighted_utility_and_worst_safety(self):
        perf = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.double)
        safe = torch.tensor(
            [[
                [0.5, 0.2],
                [0.4, -0.1],
                [0.3, 0.1],
            ]],
            dtype=torch.double,
        )
        weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.double)

        fused_perf, fused_safe = fused_single_task_observations(perf, safe, weights)

        self.assertTrue(torch.allclose(fused_perf, torch.tensor([[2.3]], dtype=torch.double)))
        self.assertTrue(torch.allclose(fused_safe, torch.tensor([[-0.1]], dtype=torch.double)))

    def test_hybrid_discontinuity_lowers_transition_safety_for_high_impact(self):
        x = torch.full((1, INPUT_DIM), 0.2, dtype=torch.double)
        x[:, 0:2] = 0.85
        x[:, 4:6] = 0.85
        x[:, 6:8] = 0.20
        x[:, 10:12] = 0.85

        _perf_smooth, safe_smooth = contact_rich_12d_torch(
            x,
            noise_std=0.0,
            hybrid_discontinuity=False,
        )
        _perf_hybrid, safe_hybrid = contact_rich_12d_torch(
            x,
            noise_std=0.0,
            hybrid_discontinuity=True,
            impact_threshold=0.55,
            impact_sharpness=80.0,
            impact_penalty=0.25,
        )

        transition_drop = safe_smooth[:, 1, :] - safe_hybrid[:, 1, :]
        self.assertTrue(torch.all(transition_drop > 0.20))


if __name__ == "__main__":
    unittest.main()
