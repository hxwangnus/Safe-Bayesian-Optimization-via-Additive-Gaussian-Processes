import argparse
import math
import os
import warnings

import numpy as np
import torch
import gpytorch
from gpytorch.utils.warnings import GPInputWarning

from device_utils import configure_torch_runtime, format_runtime, resolve_device, resolve_dtype
from multitask_safectrlbo import MultiTaskSafeCtrlBO
from safectrlbo import SafeCtrlBO


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
warnings.filterwarnings("ignore", category=GPInputWarning)


INPUT_DIM = 12
MODE_NAMES = ("free", "transition", "contact")
SAFETY_NAMES = ("force_margin", "stability_margin")
MODE_WEIGHTS = torch.tensor([0.25, 0.30, 0.45], dtype=torch.double)
BOUNDS_NP = np.array([[0.0] * INPUT_DIM, [1.0] * INPUT_DIM], dtype=np.float64)


def configure_reproducibility(seed, device):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_bounds(device, dtype):
    return torch.tensor(BOUNDS_NP, dtype=dtype, device=device)


def make_group_kernel_components(device, dtype, lengthscale=0.35, total_outputscale=1.0):
    groups = (
        (0, 1, 2, 3),       # free-space tracking and speed gains
        (4, 5, 6, 7),       # impedance stiffness/damping block
        (8, 9, 10, 11),     # force/impedance shaping block
        (0, 1, 4, 5),       # speed-stiffness contact impact interaction
        (4, 5, 6, 7, 10, 11),  # contact compliance interaction
    )
    outputscale = float(total_outputscale) / float(len(groups))
    components = []
    for dims in groups:
        kernel = gpytorch.kernels.RBFKernel(
            active_dims=tuple(int(d) for d in dims),
            ard_num_dims=len(dims),
        )
        kernel.initialize(lengthscale=torch.full((1, len(dims)), float(lengthscale)))
        scaled_kernel = gpytorch.kernels.ScaleKernel(kernel)
        scaled_kernel.initialize(outputscale=outputscale)
        components.append(scaled_kernel)
    return [component.to(device=device, dtype=dtype) for component in components]


def make_group_additive_kernel(device, dtype, lengthscale=0.35, total_outputscale=1.0):
    components = make_group_kernel_components(
        device=device,
        dtype=dtype,
        lengthscale=lengthscale,
        total_outputscale=total_outputscale,
    )
    return gpytorch.kernels.AdditiveKernel(*components).to(device=device, dtype=dtype)


def make_surrogate_kernel(surrogate, device, dtype, lengthscale=0.35, total_outputscale=1.0):
    if surrogate == "icm":
        return make_group_additive_kernel(
            device=device,
            dtype=dtype,
            lengthscale=lengthscale,
            total_outputscale=total_outputscale,
        )
    if surrogate == "lmc":
        return make_group_kernel_components(
            device=device,
            dtype=dtype,
            lengthscale=lengthscale,
            total_outputscale=total_outputscale,
        )
    raise ValueError("surrogate must be 'icm' or 'lmc'.")


def _expand_group_values(values, dtype, device):
    speed, tracking, stiffness, damping, force_gain, impedance = values
    return torch.tensor(
        [
            speed, speed,
            tracking, tracking,
            stiffness, stiffness,
            damping, damping,
            force_gain, force_gain,
            impedance, impedance,
        ],
        dtype=dtype,
        device=device,
    )


def mode_targets(dtype, device):
    return torch.stack(
        [
            _expand_group_values((0.66, 0.70, 0.38, 0.55, 0.42, 0.45), dtype, device),
            _expand_group_values((0.45, 0.56, 0.48, 0.74, 0.46, 0.56), dtype, device),
            _expand_group_values((0.35, 0.50, 0.56, 0.82, 0.50, 0.66), dtype, device),
        ],
        dim=0,
    )


def safe_anchor(dtype, device):
    return _expand_group_values((0.25, 0.30, 0.25, 0.70, 0.30, 0.35), dtype, device)


def contact_rich_12d_torch(
    x,
    noise_std=0.0,
    hybrid_discontinuity=False,
    impact_threshold=0.55,
    impact_sharpness=80.0,
    impact_penalty=0.20,
):
    """
    Synthetic mode-aware 12D controller tuning problem.

    Returns:
        perf_modes: (n, 3), one objective per mode
        safe_modes: (n, 3, 2), force/stability safety margins per mode
    """
    x = x.view(-1, INPUT_DIM)
    dtype = x.dtype
    device = x.device

    speed = x[:, 0:2].mean(dim=-1)
    tracking = x[:, 2:4].mean(dim=-1)
    stiffness = x[:, 4:6].mean(dim=-1)
    damping = x[:, 6:8].mean(dim=-1)
    force_gain = x[:, 8:10].mean(dim=-1)
    impedance = x[:, 10:12].mean(dim=-1)

    targets = mode_targets(dtype, device)
    scale = torch.tensor(
        [0.32, 0.32, 0.30, 0.30, 0.34, 0.34, 0.30, 0.30, 0.34, 0.34, 0.34, 0.34],
        dtype=dtype,
        device=device,
    )
    diff = (x.unsqueeze(1) - targets.unsqueeze(0)) / scale.view(1, 1, -1)
    perf = 1.15 - 0.38 * diff.square().mean(dim=-1)
    perf[:, 0] = perf[:, 0] + 0.08 * tracking + 0.04 * speed
    perf[:, 1] = perf[:, 1] + 0.06 * damping - 0.03 * speed
    perf[:, 2] = perf[:, 2] + 0.08 * impedance + 0.05 * damping

    free_force = 0.48 - (0.20 * speed.square() + 0.08 * stiffness.square() + 0.05 * impedance.square())
    trans_force = 0.42 - (
        0.25 * speed.square()
        + 0.30 * stiffness.square()
        + 0.20 * impedance.square()
        - 0.18 * damping
    )
    contact_force = 0.40 - (
        0.15 * speed.square()
        + 0.38 * stiffness.square()
        + 0.28 * impedance.square()
        - 0.24 * damping
    )

    free_stability = 0.45 - (0.15 * tracking + 0.10 * speed + 0.06 * stiffness - 0.12 * damping)
    trans_stability = 0.35 - (
        0.18 * tracking + 0.18 * speed + 0.22 * stiffness + 0.10 * impedance - 0.25 * damping
    )
    contact_stability = 0.34 - (
        0.14 * tracking + 0.12 * speed + 0.30 * stiffness + 0.18 * impedance - 0.30 * damping
    )

    if hybrid_discontinuity:
        impact = speed * stiffness + 0.8 * impedance - 0.7 * damping
        if impact_sharpness is None or impact_sharpness <= 0.0:
            contact_cliff = (impact > impact_threshold).to(dtype=dtype)
        else:
            contact_cliff = torch.sigmoid(float(impact_sharpness) * (impact - float(impact_threshold)))
        penalty = float(impact_penalty) * contact_cliff
        trans_force = trans_force - penalty
        trans_stability = trans_stability - penalty
        contact_force = contact_force - 0.35 * penalty
        contact_stability = contact_stability - 0.25 * penalty
        perf[:, 1] = perf[:, 1] - 0.04 * contact_cliff
        perf[:, 2] = perf[:, 2] - 0.03 * contact_cliff

    safe = torch.stack(
        [
            torch.stack([free_force, free_stability], dim=-1),
            torch.stack([trans_force, trans_stability], dim=-1),
            torch.stack([contact_force, contact_stability], dim=-1),
        ],
        dim=1,
    )

    if noise_std is not None and noise_std > 0.0:
        perf = perf + noise_std * torch.randn_like(perf)
        safe = safe + noise_std * torch.randn_like(safe)

    return perf, safe


def utility(perf_modes, mode_weights):
    weights = mode_weights.to(device=perf_modes.device, dtype=perf_modes.dtype).view(1, -1)
    return (perf_modes * weights).sum(dim=-1)


def fused_single_task_observations(perf_modes, safe_modes, mode_weights):
    fused_perf = utility(perf_modes, mode_weights).view(-1, 1)
    fused_safe = safe_modes.amin(dim=(1, 2)).view(-1, 1)
    return fused_perf, fused_safe


def is_certified_suggestion_mode(mode):
    return not str(mode).startswith("empirical_")


def sample_initial_safe_points(
    num_initial,
    run_rng,
    device,
    dtype,
    noise_std,
    max_attempts=20000,
    hybrid_discontinuity=False,
    impact_threshold=0.55,
    impact_sharpness=80.0,
    impact_penalty=0.20,
):
    anchor = safe_anchor(dtype, device)
    points = [anchor]
    perfs = []
    safes = []
    perf_anchor, safe_anchor_values = contact_rich_12d_torch(
        anchor.view(1, -1),
        noise_std=noise_std,
        hybrid_discontinuity=hybrid_discontinuity,
        impact_threshold=impact_threshold,
        impact_sharpness=impact_sharpness,
        impact_penalty=impact_penalty,
    )
    perfs.append(perf_anchor)
    safes.append(safe_anchor_values)

    attempts = 0
    while len(points) < num_initial and attempts < max_attempts:
        attempts += 1
        candidate_np = run_rng.normal(
            loc=anchor.detach().cpu().numpy(),
            scale=0.06,
            size=(INPUT_DIM,),
        )
        candidate_np = np.clip(candidate_np, 0.0, 1.0)
        candidate = torch.tensor(candidate_np, dtype=dtype, device=device).view(1, -1)
        true_perf, true_safe = contact_rich_12d_torch(
            candidate,
            noise_std=0.0,
            hybrid_discontinuity=hybrid_discontinuity,
            impact_threshold=impact_threshold,
            impact_sharpness=impact_sharpness,
            impact_penalty=impact_penalty,
        )
        if torch.all(true_safe >= 0.0):
            if noise_std is not None and noise_std > 0.0:
                perf, safe = contact_rich_12d_torch(
                    candidate,
                    noise_std=noise_std,
                    hybrid_discontinuity=hybrid_discontinuity,
                    impact_threshold=impact_threshold,
                    impact_sharpness=impact_sharpness,
                    impact_penalty=impact_penalty,
                )
            else:
                perf, safe = true_perf, true_safe
            points.append(candidate.squeeze(0))
            perfs.append(perf)
            safes.append(safe)

    if len(points) < num_initial:
        raise RuntimeError(f"Could not sample {num_initial} safe initial points.")

    return torch.stack(points, dim=0), torch.cat(perfs, dim=0), torch.cat(safes, dim=0)


@torch.no_grad()
def estimate_best_feasible_utility(
    num_samples,
    seed,
    device,
    dtype,
    hybrid_discontinuity=False,
    impact_threshold=0.55,
    impact_sharpness=80.0,
    impact_penalty=0.20,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0 if seed is None else int(seed) + 137)
    X = torch.rand((num_samples, INPUT_DIM), generator=generator, dtype=dtype, device=device)
    perf, safe = contact_rich_12d_torch(
        X,
        noise_std=0.0,
        hybrid_discontinuity=hybrid_discontinuity,
        impact_threshold=impact_threshold,
        impact_sharpness=impact_sharpness,
        impact_penalty=impact_penalty,
    )
    safe_mask = torch.all(safe >= 0.0, dim=(1, 2))
    if not torch.any(safe_mask):
        return float("nan")
    util = utility(perf, MODE_WEIGHTS.to(device=device, dtype=dtype))
    return float(util[safe_mask].max().item())


def run_experiment(
    iterations=40,
    num_candidates=4096,
    num_initial=6,
    switch_time=6,
    safe_retry_radius=0.06,
    rkhs_bound=0.12,
    noise_bound=0.01,
    delta=0.05,
    noise_std=1e-4,
    init_training_iter=15,
    train_hypers_every=None,
    training_iter=0,
    hybrid_discontinuity=False,
    impact_threshold=0.55,
    impact_sharpness=80.0,
    impact_penalty=0.20,
    method="mode-aware",
    surrogate="lmc",
    device=None,
    dtype=torch.float64,
    seed=0,
):
    device = resolve_device(device or "auto")
    dtype = resolve_dtype(dtype)
    configure_reproducibility(seed, device)
    run_rng = np.random.default_rng(seed)
    if method not in {"mode-aware", "fused-single-task"}:
        raise ValueError("method must be 'mode-aware' or 'fused-single-task'.")
    if surrogate not in {"icm", "lmc"}:
        raise ValueError("surrogate must be 'icm' or 'lmc'.")

    bounds = make_bounds(device, dtype)
    X0, Yf0, Yg0 = sample_initial_safe_points(
        num_initial=num_initial,
        run_rng=run_rng,
        device=device,
        dtype=dtype,
        noise_std=noise_std,
        hybrid_discontinuity=hybrid_discontinuity,
        impact_threshold=impact_threshold,
        impact_sharpness=impact_sharpness,
        impact_penalty=impact_penalty,
    )

    if method == "mode-aware":
        base_kernel = make_surrogate_kernel(surrogate=surrogate, device=device, dtype=dtype)
        algo = MultiTaskSafeCtrlBO(
            init_X=X0,
            init_Y_perf=Yf0,
            init_Y_safe=Yg0,
            bounds=bounds,
            base_kernel=base_kernel,
            safety_threshold=torch.zeros((len(MODE_NAMES), len(SAFETY_NAMES)), dtype=dtype, device=device),
            mode_names=MODE_NAMES,
            mode_weights=MODE_WEIGHTS.to(device=device, dtype=dtype),
            switch_time=switch_time,
            tau=0.08,
            device=device,
            init_training_iter=init_training_iter,
            likelihood_noise=max(float(noise_std) ** 2, 1e-6),
            sobol_seed=seed,
            safe_retry_radius=safe_retry_radius,
            rkhs_bound=rkhs_bound,
            noise_bound=noise_bound,
            delta=delta,
            task_rank=2,
            expansion_uncertainty="safety",
            multitask_kernel=surrogate,
        )
    else:
        base_kernel = make_group_additive_kernel(device=device, dtype=dtype)
        fused_Yf0, fused_Yg0 = fused_single_task_observations(
            Yf0,
            Yg0,
            MODE_WEIGHTS.to(device=device, dtype=dtype),
        )
        # SafeCtrlBO still interprets switch_time as total observations. Shift
        # it here so the baseline gets the same number of BO expansion steps.
        fused_switch_time = max(int(num_initial) + int(switch_time) - 1, -1)
        algo = SafeCtrlBO(
            init_X=X0,
            init_Y_perf=fused_Yf0,
            init_Y_safe=fused_Yg0,
            bounds=bounds,
            base_kernel=base_kernel,
            safety_threshold=torch.tensor(0.0, dtype=dtype, device=device),
            switch_time=fused_switch_time,
            tau=0.08,
            device=device,
            init_training_iter=init_training_iter,
            likelihood_noise=max(float(noise_std) ** 2, 1e-6),
            sobol_seed=seed,
            safe_retry_radius=safe_retry_radius,
            rkhs_bound=rkhs_bound,
            noise_bound=noise_bound,
            delta=delta,
            expansion_uncertainty="safety",
        )

    initial_utility = float(utility(Yf0, MODE_WEIGHTS.to(device=device, dtype=dtype)).max().item())
    best_safe_utility = initial_utility
    safety_violations = 0
    severe_violations = 0
    certified_decision_count = 0
    certified_false_safe_count = 0
    worst_mode_constraint_counts = {
        f"{mode_name}/{constraint_name}": 0
        for mode_name in MODE_NAMES
        for constraint_name in SAFETY_NAMES
    }

    for t in range(iterations):
        x_next, mode, _sets = algo.suggest(num_candidates=num_candidates)
        certified_decision = is_certified_suggestion_mode(mode)
        if certified_decision:
            certified_decision_count += 1
        y_perf, y_safe = contact_rich_12d_torch(
            x_next,
            noise_std=noise_std,
            hybrid_discontinuity=hybrid_discontinuity,
            impact_threshold=impact_threshold,
            impact_sharpness=impact_sharpness,
            impact_penalty=impact_penalty,
        )
        true_perf, true_safe = contact_rich_12d_torch(
            x_next,
            noise_std=0.0,
            hybrid_discontinuity=hybrid_discontinuity,
            impact_threshold=impact_threshold,
            impact_sharpness=impact_sharpness,
            impact_penalty=impact_penalty,
        )
        is_safe = bool(torch.all(true_safe >= 0.0).item())
        min_margin = float(true_safe.min().item())
        worst_flat_idx = int(torch.argmin(true_safe.view(-1)).item())
        worst_mode_idx = worst_flat_idx // len(SAFETY_NAMES)
        worst_constraint_idx = worst_flat_idx % len(SAFETY_NAMES)
        worst_key = f"{MODE_NAMES[worst_mode_idx]}/{SAFETY_NAMES[worst_constraint_idx]}"
        if not is_safe:
            safety_violations += 1
            if certified_decision:
                certified_false_safe_count += 1
            worst_mode_constraint_counts[worst_key] += 1
        if min_margin < -0.05:
            severe_violations += 1

        util_value = float(utility(true_perf, MODE_WEIGHTS.to(device=device, dtype=dtype)).item())
        if is_safe and util_value > best_safe_utility:
            best_safe_utility = util_value

        if method == "mode-aware":
            algo.observe(
                x_new=x_next,
                y_perf_new=y_perf,
                y_safe_new=y_safe,
                train_hypers_every=train_hypers_every,
                training_iter=training_iter,
            )
        else:
            fused_y_perf, fused_y_safe = fused_single_task_observations(
                y_perf,
                y_safe,
                MODE_WEIGHTS.to(device=device, dtype=dtype),
            )
            algo.observe(
                x_new=x_next,
                y_perf_new=fused_y_perf,
                y_safe_new=fused_y_safe,
                train_hypers_every=train_hypers_every,
                training_iter=training_iter,
            )

        print(
            f"iter={t + 1:03d} mode={mode:<24} "
            f"certified={certified_decision} safe={is_safe} min_margin={min_margin:+.4f} "
            f"worst={worst_key:<27} "
            f"best_safe_utility={best_safe_utility:.4f}"
        )

    best_estimate = estimate_best_feasible_utility(
        num_samples=20000,
        seed=seed,
        device=device,
        dtype=dtype,
        hybrid_discontinuity=hybrid_discontinuity,
        impact_threshold=impact_threshold,
        impact_sharpness=impact_sharpness,
        impact_penalty=impact_penalty,
    )

    return {
        "initial_utility": initial_utility,
        "method": method,
        "surrogate": surrogate if method == "mode-aware" else "single-task",
        "best_safe_utility": best_safe_utility,
        "estimated_best_feasible_utility": best_estimate,
        "improvement": best_safe_utility - initial_utility,
        "safety_violations": safety_violations,
        "violation_rate": safety_violations / float(max(iterations, 1)),
        "certified_decision_count": certified_decision_count,
        "certified_false_safe_count": certified_false_safe_count,
        "certified_false_safe_rate": certified_false_safe_count / float(max(certified_decision_count, 1)),
        "false_safe_count": certified_false_safe_count,
        "false_safe_rate": certified_false_safe_count / float(max(certified_decision_count, 1)),
        "severe_violations": severe_violations,
        "unsafe_worst_mode_constraint_counts": worst_mode_constraint_counts,
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--method", type=str, default="mode-aware", choices=["mode-aware", "fused-single-task"])
    parser.add_argument("--surrogate", type=str, default="lmc", choices=["icm", "lmc"])
    parser.add_argument("--num-candidates", type=int, default=4096)
    parser.add_argument("--num-initial", type=int, default=6)
    parser.add_argument("--switch-time", type=int, default=6)
    parser.add_argument("--rkhs-bound", type=float, default=0.12)
    parser.add_argument("--noise-bound", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=1e-4)
    parser.add_argument("--init-training-iter", type=int, default=15)
    parser.add_argument("--train-hypers-every", type=int, default=None)
    parser.add_argument("--training-iter", type=int, default=0)
    parser.add_argument("--hybrid-discontinuity", action="store_true")
    parser.add_argument("--impact-threshold", type=float, default=0.55)
    parser.add_argument("--impact-sharpness", type=float, default=80.0)
    parser.add_argument("--impact-penalty", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = configure_torch_runtime(args.device)
    dtype = resolve_dtype(args.dtype)
    print(f"Running 12D contact-mode Safe BO with {format_runtime(device, dtype)}")
    print(f"Modes: {', '.join(MODE_NAMES)}")
    print(f"Safety constraints: {', '.join(SAFETY_NAMES)}")
    print(f"Method: {args.method}")
    print(f"Surrogate: {args.surrogate if args.method == 'mode-aware' else 'single-task'}")
    print(f"Hybrid discontinuity: {args.hybrid_discontinuity}")
    print(f"Seed: {args.seed}")

    summary = run_experiment(
        iterations=args.iterations,
        num_candidates=args.num_candidates,
        num_initial=args.num_initial,
        switch_time=args.switch_time,
        rkhs_bound=args.rkhs_bound,
        noise_bound=args.noise_bound,
        delta=args.delta,
        noise_std=args.noise_std,
        init_training_iter=args.init_training_iter,
        train_hypers_every=args.train_hypers_every,
        training_iter=args.training_iter,
        hybrid_discontinuity=args.hybrid_discontinuity,
        impact_threshold=args.impact_threshold,
        impact_sharpness=args.impact_sharpness,
        impact_penalty=args.impact_penalty,
        method=args.method,
        surrogate=args.surrogate,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    print("")
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
