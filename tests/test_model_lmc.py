import unittest

import gpytorch
import torch

from model import build_mode_lmc_gp, fit_mode_task_gp, iter_multitask_kernels


def make_component(active_dims):
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(active_dims=active_dims, ard_num_dims=len(active_dims))
    )
    kernel.initialize(outputscale=0.5)
    kernel.base_kernel.initialize(lengthscale=torch.ones(1, len(active_dims), dtype=torch.double))
    return kernel


class ModeLMCGPTests(unittest.TestCase):
    def test_build_mode_lmc_gp_splits_additive_kernel_components(self):
        train_X = torch.tensor(
            [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]],
            dtype=torch.double,
        )
        train_Y = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.4, 0.2]],
            dtype=torch.double,
        )
        additive = gpytorch.kernels.AdditiveKernel(make_component((0, 1)), make_component((2,)))

        model, _likelihood, _mll = build_mode_lmc_gp(
            train_X,
            train_Y,
            additive,
            num_modes=3,
            noise=1e-4,
            task_rank=1,
        )

        self.assertEqual(len(list(iter_multitask_kernels(model.covar_module))), 2)

    def test_build_mode_lmc_gp_uses_one_multitask_kernel_per_component(self):
        train_X = torch.tensor(
            [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]],
            dtype=torch.double,
        )
        train_Y = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.4, 0.2]],
            dtype=torch.double,
        )
        components = [make_component((0, 1)), make_component((2,))]

        model, likelihood, mll = build_mode_lmc_gp(
            train_X,
            train_Y,
            components,
            num_modes=3,
            noise=1e-4,
            task_rank=2,
        )

        multitask_kernels = list(iter_multitask_kernels(model.covar_module))
        self.assertEqual(len(multitask_kernels), 2)
        self.assertIsNotNone(mll)

        model.eval()
        likelihood.eval()
        posterior = model(train_X + 0.01)
        self.assertEqual(tuple(posterior.mean.shape), (3, 3))

    def test_fit_mode_task_gp_handles_lmc_kernel_freezing(self):
        train_X = torch.tensor(
            [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]],
            dtype=torch.double,
        )
        train_Y = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.4, 0.2]],
            dtype=torch.double,
        )
        model, likelihood, mll = build_mode_lmc_gp(
            train_X,
            train_Y,
            [make_component((0, 1)), make_component((2,))],
            num_modes=3,
            noise=1e-4,
            task_rank=1,
        )

        fit_mode_task_gp(
            model,
            likelihood,
            mll,
            training_iter=1,
            train_data_kernel=False,
            train_task_covar=True,
            train_mean=False,
            train_noise=True,
        )

        for multitask_kernel in iter_multitask_kernels(model.covar_module):
            self.assertTrue(all(not p.requires_grad for p in multitask_kernel.data_covar_module.parameters()))
            self.assertTrue(any(p.requires_grad for p in multitask_kernel.task_covar_module.parameters()))


if __name__ == "__main__":
    unittest.main()
