# camelback.py
from __future__ import print_function, division, absolute_import

import sys
import os
import tempfile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow duplicated OpenMP runtime
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np

# Optional: limit threads to avoid OMP issues / oversubscription
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# --- Matplotlib backend setup ---
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend for WSL/headless environments
import matplotlib.pyplot as plt

import gpytorch

from safectrlbo import SafeCtrlBO

# ----------------------------------------
# Global settings
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Measurement noise (for GP likelihood)
noise_var = 0.01 ** 2

# Bounds on the input variables: x in [-2,2], y in [-1,1]
bounds_np = np.array([[-2.0, -1.0],
                      [ 2.0,  1.0]])          # shape (2, d)
bounds = torch.tensor(bounds_np, dtype=torch.double, device=device)

# ----------------------------------------
# Camelback-like objective (maximization)
# y(x) = max(-f(x), -2.5), same as your original code
# ----------------------------------------
def camelback_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., 2) tensor
    returns: (...,) tensor
    """
    x = x.view(-1, 2)
    xx = x[:, 0]
    yy = x[:, 1]
    f = (4.0 - 2.1 * xx**2 + (xx**4) / 3.0) * (xx**2) \
        + xx * yy \
        + (-4.0 + 4.0 * (yy**2)) * (yy**2)
    # maximization of -f, clipped at -2.5
    y = torch.maximum(-f, torch.tensor(-2.5, dtype=x.dtype, device=x.device))
    return y  # shape (batch,)

# Global optimum of the six-hump camel function under maximization of -f.
GLOBAL_OPT = 1.031628453489877
PLOT_FLOOR = 1e-12

# ----------------------------------------
# Build base kernel: k1 + k2 + k1*k2
# ----------------------------------------
def make_additive_kernel_k1_k2_k1k2():
    """
    Build a kernel k(x) = k1(x_0) + k2(x_1) + k1(x_0)*k2(x_1),
    where k1, k2 are 1D RBF kernels on dim 0 and dim 1 respectively.

    This matches the GPy setup:
        k1: variance=2, lengthscale=1
        k2: variance=1, lengthscale=1
        kernel = k1 + k2 + k1*k2

    In GPyTorch:
        - variance is represented as ScaleKernel.outputscale
        - lengthscale is inside RBFKernel
    """
    # RBF on dim 0
    rbf0 = gpytorch.kernels.RBFKernel(active_dims=(0,))
    rbf0.initialize(lengthscale=1.0)
    k1 = gpytorch.kernels.ScaleKernel(rbf0)
    k1.initialize(outputscale=2.0)   # variance ~ 2.0

    # RBF on dim 1
    rbf1 = gpytorch.kernels.RBFKernel(active_dims=(1,))
    rbf1.initialize(lengthscale=1.0)
    k2 = gpytorch.kernels.ScaleKernel(rbf1)
    k2.initialize(outputscale=1.0)   # variance ~ 1.0

    # Interaction term k1 * k2 (product kernel)
    k12 = gpytorch.kernels.ProductKernel(k1, k2)

    # Additive kernel: k1 + k2 + k1*k2
    base_kernel = gpytorch.kernels.AdditiveKernel(k1, k2, k12)
    return base_kernel.to(device=device, dtype=torch.double)

# ----------------------------------------
# Main experiment: SafeCtrlBO on Camelback with NO explicit safety
# ----------------------------------------
def run_experiment(
    num_runs: int = 100,
    iterations: int = 100,
    num_candidates: int = 8192,
):
    """
    Run multiple BO runs with SafeCtrlBO on the Camelback-like function.

    Here we use the "no safety" mode of SafeCtrlBO:
    - init_Y_safe=None, safety_threshold=None
    - observe() is called without y_safe_new
    SafeCtrlBO internally falls back to unconstrained BO (S=B=all points).
    """
    all_simple_regret = np.zeros((iterations, num_runs), dtype=float)

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        # Random initial point x0 ~ U([-2,2] x [-1,1])
        x0_np = np.random.uniform(low=[-2.0, -1.0],
                                  high=[ 2.0,  1.0],
                                  size=(1, 2))
        x0 = torch.tensor(x0_np, dtype=torch.double, device=device)  # (1,2)

        # Initial performance
        y0 = camelback_torch(x0)        # (1,)
        y0 = y0.view(-1, 1)             # (1,1)

        # NO safety data: use_safety=False inside SafeCtrlBO
        init_Y_safe = None
        safety_threshold = None

        # Build additive kernel k1 + k2 + k1*k2
        base_kernel = make_additive_kernel_k1_k2_k1k2()

        # Initialize SafeCtrlBO in "no safety" mode
        algo = SafeCtrlBO(
            init_X=x0,
            init_Y_perf=y0,
            init_Y_safe=init_Y_safe,   # None => unconstrained mode
            bounds=bounds,
            base_kernel=base_kernel,
            safety_threshold=safety_threshold,  # None => no constraint
            switch_time=0,       # pure exploitation/exploration split at t=0 (like old code)
            beta_fn=None,
            tau=0.1,             # irrelevant when there is no safety
            device=device,
            init_training_iter=0 # keep kernel hyperparameters fixed
        )

        # Track best value and simple regret
        y_best = y0.item()
        simple_regrets = []

        for t in range(iterations):
            # print progress in-place
            sys.stdout.write(f"\rRun {run + 1}/{num_runs} - Iteration {t + 1}/{iterations}")
            sys.stdout.flush()

            # Suggest next x
            x_next, mode, sets = algo.suggest(num_candidates=num_candidates)  # x_next: (1,2)

            # Evaluate objective
            y_next = camelback_torch(x_next)  # (1,)
            y_next_val = y_next.item()
            y_next_tensor = y_next.view(-1, 1)  # (1,1)

            # Update best value
            if y_next_val > y_best:
                y_best = y_next_val

            # Simple regret wrt known optimum
            simple_regret = max(GLOBAL_OPT - y_best, 0.0)
            simple_regrets.append(simple_regret)
            all_simple_regret[t, run] = simple_regret

            # Add new data (no safety)
            algo.observe(
                x_new=x_next,
                y_perf_new=y_next_tensor,
                # y_safe_new=None (default)
                train_hypers_every=None,  # no online hyper-optimization
                training_iter=0,
            )

        final_simple_regret = simple_regrets[-1]
        print(f"\nFinal Simple Regret for Run {run + 1}: {final_simple_regret:.4f}")

    return all_simple_regret


def main():
    num_runs = 100
    iterations = 100

    all_simple_regret_matrix = run_experiment(
        num_runs=num_runs,
        iterations=iterations,
        num_candidates=16384,
    )

    # compute mean and std over runs
    mean_simple_regret = np.mean(all_simple_regret_matrix, axis=1)
    std_simple_regret = np.std(all_simple_regret_matrix, axis=1)

    # plot the statistical result
    print("Plotting simple regret curve...")
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(1, iterations + 1)
    mean_curve = np.clip(mean_simple_regret, PLOT_FLOOR, None)
    lower_band = np.clip(mean_simple_regret - std_simple_regret, PLOT_FLOOR, None)
    upper_band = np.clip(mean_simple_regret + std_simple_regret, PLOT_FLOOR, None)

    plt.plot(x_axis, mean_curve, label='Mean Simple Regret')
    plt.fill_between(
        x_axis,
        lower_band,
        upper_band,
        alpha=0.3,
        label='Standard Deviation'
    )
    plt.yscale('log')
    plt.xlabel('Optimization Step')
    plt.ylabel('Simple Regret')
    plt.title('Simple Regret over Optimization Steps (SafeCtrlBO, k1 + k2 + k1*k2 kernel, no safety)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = "camelback_simple_regret.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {out_path}")
    print("Done plotting.")


if __name__ == "__main__":
    main()
