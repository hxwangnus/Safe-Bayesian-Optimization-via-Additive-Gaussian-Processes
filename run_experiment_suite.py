import argparse
import contextlib
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from camelback import (
    PLOT_FLOOR as CAMELBACK_PLOT_FLOOR,
    run_experiment as run_camelback_experiment,
    summarize_regret as summarize_camelback_regret,
)
from contact_mode_benchmark import run_experiment as run_contact_experiment
from device_utils import configure_torch_runtime, resolve_dtype
from hartmann import (
    PLOT_FLOOR as HARTMANN_PLOT_FLOOR,
    run_experiment as run_hartmann_experiment,
    summarize_regret as summarize_hartmann_regret,
)


CONTACT_METHODS = {
    "single-task": {
        "label": "Single-task fused",
        "method": "fused-single-task",
        "surrogate": "lmc",
    },
    "icm": {
        "label": "Mode-aware ICM",
        "method": "mode-aware",
        "surrogate": "icm",
    },
    "lmc": {
        "label": "Mode-aware LMC",
        "method": "mode-aware",
        "surrogate": "lmc",
    },
}


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def run_with_optional_log(func, log_path, quiet, *args, **kwargs):
    if not quiet:
        return func(*args, **kwargs)

    with open(log_path, "a", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file):
            return func(*args, **kwargs)


def format_mean_std(values, digits=4):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.{digits}f} +/- {values.std(ddof=0):.{digits}f}"


def plot_regret_curve(regret_matrix, summarize_fn, success_threshold, plot_floor, title, out_path):
    mean_regret, median_regret, _std_regret, q25_regret, q75_regret, success_rate = summarize_fn(
        regret_matrix,
        success_threshold=success_threshold,
    )
    x_axis = np.arange(1, regret_matrix.shape[0] + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, np.clip(mean_regret, plot_floor, None), label="Mean simple regret")
    plt.plot(x_axis, np.clip(median_regret, plot_floor, None), "--", label="Median simple regret")
    plt.fill_between(
        x_axis,
        np.clip(q25_regret, plot_floor, None),
        np.clip(q75_regret, plot_floor, None),
        alpha=0.25,
        label="Interquartile range",
    )
    plt.yscale("log")
    plt.xlabel("Optimization step")
    plt.ylabel("Simple regret")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    return {
        "final_mean_regret": float(mean_regret[-1]),
        "final_median_regret": float(median_regret[-1]),
        "final_success_rate": float(success_rate[-1]),
    }


def run_single_task_suite(args, out_dir, device, dtype, quiet):
    summaries = {}
    log_path = out_dir / "single_task_runs.log"

    if not args.skip_camelback:
        camelback_regret = run_with_optional_log(
            run_camelback_experiment,
            log_path,
            quiet,
            num_runs=args.camelback_num_runs,
            iterations=args.camelback_iterations,
            num_candidates=args.camelback_num_candidates,
            switch_time=args.camelback_switch_time,
            beta_mode=args.camelback_beta_mode,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )
        summaries["camelback"] = plot_regret_curve(
            camelback_regret,
            summarize_camelback_regret,
            args.camelback_success_threshold,
            CAMELBACK_PLOT_FLOOR,
            "Camelback simple regret",
            out_dir / "camelback_simple_regret.png",
        )

    if not args.skip_hartmann:
        hartmann_regret, hartmann_violations = run_with_optional_log(
            run_hartmann_experiment,
            log_path,
            quiet,
            num_runs=args.hartmann_num_runs,
            iterations=args.hartmann_iterations,
            num_candidates=args.hartmann_num_candidates,
            d_effective=args.hartmann_d_effective,
            lengthscale=args.hartmann_lengthscale,
            total_outputscale=args.hartmann_total_outputscale,
            safety_threshold=args.hartmann_safety_threshold,
            tau=args.hartmann_tau,
            switch_time=args.hartmann_switch_time,
            safe_retry_radius=args.hartmann_safe_retry_radius,
            noise_std=args.hartmann_noise_std,
            rkhs_bound=args.hartmann_rkhs_bound,
            noise_bound=args.hartmann_noise_bound,
            delta=args.hartmann_delta,
            expansion_uncertainty=args.hartmann_expansion_uncertainty,
            device=device,
            dtype=dtype,
            seed=args.seed,
            x0_file=args.hartmann_x0_file,
            max_init_attempts=args.hartmann_max_init_attempts,
        )
        summaries["hartmann"] = plot_regret_curve(
            hartmann_regret,
            summarize_hartmann_regret,
            args.hartmann_success_threshold,
            HARTMANN_PLOT_FLOOR,
            f"Hartmann6D simple regret (d_effective={args.hartmann_d_effective})",
            out_dir / "hartmann_simple_regret.png",
        )
        total_decisions = args.hartmann_num_runs * args.hartmann_iterations
        summaries["hartmann"]["total_violations"] = int(np.sum(hartmann_violations))
        summaries["hartmann"]["violation_rate"] = float(np.sum(hartmann_violations) / max(total_decisions, 1))

    return summaries


def parse_contact_seeds(args):
    if args.contact_seeds:
        return list(args.contact_seeds)
    return list(range(args.num_contact_seeds))


def run_contact_suite(args, out_dir, device, dtype, quiet):
    seeds = parse_contact_seeds(args)
    selected_methods = args.contact_methods
    log_path = out_dir / "contact_runs.log"
    raw_results = {}
    aggregate_rows = []

    for method_key in selected_methods:
        config = CONTACT_METHODS[method_key]
        label = config["label"]
        method_results = []
        print(f"Running contact suite: {label} over seeds {seeds}")

        for seed in seeds:
            summary = run_with_optional_log(
                run_contact_experiment,
                log_path,
                quiet,
                iterations=args.contact_iterations,
                num_candidates=args.contact_num_candidates,
                num_initial=args.contact_num_initial,
                switch_time=args.contact_switch_time,
                safe_retry_radius=args.contact_safe_retry_radius,
                rkhs_bound=args.contact_rkhs_bound,
                noise_bound=args.contact_noise_bound,
                delta=args.contact_delta,
                noise_std=args.contact_noise_std,
                init_training_iter=args.contact_init_training_iter,
                train_hypers_every=args.contact_train_hypers_every,
                training_iter=args.contact_training_iter,
                hybrid_discontinuity=args.hybrid_discontinuity,
                impact_threshold=args.impact_threshold,
                impact_sharpness=args.impact_sharpness,
                impact_penalty=args.impact_penalty,
                method=config["method"],
                surrogate=config["surrogate"],
                task_rank=args.contact_task_rank,
                device=device,
                dtype=dtype,
                seed=seed,
                verbose=not quiet,
            )
            summary["seed"] = seed
            summary["label"] = label
            method_results.append(summary)

        raw_results[method_key] = method_results
        improvements = np.asarray([item["improvement"] for item in method_results], dtype=float)
        utilities = np.asarray([item["best_safe_utility"] for item in method_results], dtype=float)
        violations = np.asarray([item["safety_violations"] for item in method_results], dtype=float)
        severe = np.asarray([item["severe_violations"] for item in method_results], dtype=float)
        false_safe = np.asarray([item["certified_false_safe_count"] for item in method_results], dtype=float)
        total_decisions = len(method_results) * args.contact_iterations
        aggregate_rows.append(
            {
                "method": label,
                "final_improvement": format_mean_std(improvements),
                "best_safe_utility": format_mean_std(utilities),
                "violations": int(violations.sum()),
                "violation_rate": float(violations.sum() / max(total_decisions, 1)),
                "severe_violations": int(severe.sum()),
                "false_safe": int(false_safe.sum()),
            }
        )

    plot_contact_curves(raw_results, aggregate_rows, out_dir, args.contact_iterations)
    write_contact_outputs(raw_results, aggregate_rows, out_dir)
    return {"seeds": seeds, "rows": aggregate_rows, "raw": raw_results}


def plot_mean_band(x_axis, traces, label):
    traces = np.asarray(traces, dtype=float)
    mean = traces.mean(axis=0)
    std = traces.std(axis=0)
    plt.plot(x_axis, mean, label=label)
    plt.fill_between(x_axis, mean - std, mean + std, alpha=0.16)


def plot_contact_curves(raw_results, aggregate_rows, out_dir, iterations):
    x_axis = np.arange(1, iterations + 1)

    plt.figure(figsize=(10, 6))
    for method_key, method_results in raw_results.items():
        label = CONTACT_METHODS[method_key]["label"]
        traces = []
        for item in method_results:
            trace = np.asarray(item["best_safe_utility_trace"], dtype=float)
            traces.append(trace - float(item["initial_utility"]))
        plot_mean_band(x_axis, traces, label)
    plt.xlabel("Optimization step")
    plt.ylabel("Best safe utility improvement")
    plt.title("12D contact benchmark: optimization efficiency")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "contact_best_safe_utility_improvement.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    for method_key, method_results in raw_results.items():
        label = CONTACT_METHODS[method_key]["label"]
        traces = [item["cumulative_violation_trace"] for item in method_results]
        plot_mean_band(x_axis, traces, label)
    plt.xlabel("Optimization step")
    plt.ylabel("Mean cumulative safety violations per seed")
    plt.title("12D contact benchmark: safety violations")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "contact_cumulative_violations.png", dpi=180, bbox_inches="tight")
    plt.close()

    table_columns = [
        "Method",
        "Improvement",
        "Best safe utility",
        "Viol.",
        "Viol. rate",
        "Severe",
        "False-safe",
    ]
    table_data = [
        [
            row["method"],
            row["final_improvement"],
            row["best_safe_utility"],
            str(row["violations"]),
            f"{100.0 * row['violation_rate']:.2f}%",
            str(row["severe_violations"]),
            str(row["false_safe"]),
        ]
        for row in aggregate_rows
    ]

    fig_height = 1.1 + 0.45 * max(len(table_data), 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=table_columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    ax.set_title("12D contact benchmark summary", pad=14, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "contact_summary_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_contact_outputs(raw_results, aggregate_rows, out_dir):
    with open(out_dir / "contact_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(
            {"rows": aggregate_rows, "raw": json_ready(raw_results)},
            file_obj,
            indent=2,
        )

    with open(out_dir / "contact_summary.csv", "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "method",
                "final_improvement",
                "best_safe_utility",
                "violations",
                "violation_rate",
                "severe_violations",
                "false_safe",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Run and plot SafeCtrlBO public-repo experiments.")
    parser.add_argument("--output-dir", type=str, default="results/public_experiments")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-subruns", action="store_true")

    parser.add_argument("--skip-single-task", action="store_true")
    parser.add_argument("--skip-camelback", action="store_true")
    parser.add_argument("--skip-hartmann", action="store_true")
    parser.add_argument("--skip-contact", action="store_true")

    parser.add_argument("--camelback-num-runs", type=int, default=100)
    parser.add_argument("--camelback-iterations", type=int, default=150)
    parser.add_argument("--camelback-num-candidates", type=int, default=16384)
    parser.add_argument("--camelback-switch-time", type=int, default=0)
    parser.add_argument("--camelback-beta-mode", choices=["legacy", "paper"], default="legacy")
    parser.add_argument("--camelback-success-threshold", type=float, default=1e-4)

    parser.add_argument("--hartmann-num-runs", type=int, default=10)
    parser.add_argument("--hartmann-iterations", type=int, default=100)
    parser.add_argument("--hartmann-num-candidates", type=int, default=1024)
    parser.add_argument("--hartmann-d-effective", type=int, default=6)
    parser.add_argument("--hartmann-lengthscale", type=float, default=1.0)
    parser.add_argument("--hartmann-total-outputscale", type=float, default=1.0)
    parser.add_argument("--hartmann-noise-std", type=float, default=1e-4)
    parser.add_argument("--hartmann-safety-threshold", type=float, default=0.3)
    parser.add_argument("--hartmann-tau", type=float, default=0.2)
    parser.add_argument("--hartmann-switch-time", type=int, default=5)
    parser.add_argument("--hartmann-safe-retry-radius", type=float, default=0.05)
    parser.add_argument("--hartmann-rkhs-bound", type=float, default=2.0)
    parser.add_argument("--hartmann-noise-bound", type=float, default=None)
    parser.add_argument("--hartmann-delta", type=float, default=0.05)
    parser.add_argument("--hartmann-expansion-uncertainty", choices=["safety", "objective"], default="safety")
    parser.add_argument("--hartmann-x0-file", type=str, default=None)
    parser.add_argument("--hartmann-max-init-attempts", type=int, default=10000)
    parser.add_argument("--hartmann-success-threshold", type=float, default=1e-2)

    parser.add_argument("--num-contact-seeds", type=int, default=10)
    parser.add_argument("--contact-seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--contact-methods",
        nargs="+",
        choices=sorted(CONTACT_METHODS),
        default=["single-task", "icm", "lmc"],
    )
    parser.add_argument("--contact-iterations", type=int, default=100)
    parser.add_argument("--contact-num-candidates", type=int, default=1024)
    parser.add_argument("--contact-num-initial", type=int, default=6)
    parser.add_argument("--contact-switch-time", type=int, default=4)
    parser.add_argument("--contact-safe-retry-radius", type=float, default=0.06)
    parser.add_argument("--contact-rkhs-bound", type=float, default=0.12)
    parser.add_argument("--contact-noise-bound", type=float, default=0.01)
    parser.add_argument("--contact-delta", type=float, default=0.05)
    parser.add_argument("--contact-noise-std", type=float, default=1e-4)
    parser.add_argument("--contact-init-training-iter", type=int, default=15)
    parser.add_argument("--contact-train-hypers-every", type=int, default=10)
    parser.add_argument("--contact-training-iter", type=int, default=2)
    parser.add_argument("--contact-task-rank", type=int, default=2)
    parser.add_argument("--hybrid-discontinuity", action="store_true")
    parser.add_argument("--impact-threshold", type=float, default=0.45)
    parser.add_argument("--impact-sharpness", type=float, default=80.0)
    parser.add_argument("--impact-penalty", type=float, default=0.30)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = configure_torch_runtime(args.device)
    dtype = resolve_dtype(args.dtype)
    quiet = not args.verbose_subruns
    summaries = {}

    if not args.skip_single_task:
        summaries["single_task"] = run_single_task_suite(args, out_dir, device, dtype, quiet)

    if not args.skip_contact:
        summaries["contact"] = run_contact_suite(args, out_dir, device, dtype, quiet)

    with open(out_dir / "suite_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(json_ready(summaries), file_obj, indent=2)

    print(f"Experiment suite outputs written to {out_dir}")


if __name__ == "__main__":
    main()
