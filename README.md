# SafeCtrlBO

GPyTorch implementation of SafeCtrlBO for safe Bayesian optimization in two
maintained workflows:

1. **Single-task SafeCtrlBO**
   - `benchmarks/camelback.py`: 2D Camelback benchmark in unconstrained BO mode.
   - `benchmarks/hartmann.py`: 6D Hartmann safe BO benchmark with safety-violation
     reporting.
2. **Mode-aware multi-task SafeCtrlBO**
   - `benchmarks/contact_mode_benchmark.py`: 12D synthetic robot contact
     benchmark with `free`, `transition`, and `contact` modes.
   - Supports mode-aware LMC and ICM surrogates, plus a fused single-task
     baseline for comparison.

The historical kernel-search files are kept as an archive, not as the primary
workflow. In particular, `archive/kernel_search/selectKernel.py` and
`archive/kernel_search/gantry_data1.csv` document an earlier DARTS-style
kernel-search idea. `src/safectrlbo/model.py` is not a user-facing experiment
script; it is an internal GP model helper module imported by the current
optimizers.

## Supported Workflows

### Single-Task SafeCtrlBO

`src/safectrlbo/safectrlbo.py` implements the single-task optimizer. It can run
in two modes:

- **Unconstrained BO** when `init_Y_safe=None` and `safety_threshold=None`.
  This is how `benchmarks/camelback.py` is configured.
- **Safe BO** when safety observations and a threshold are provided. This is
  how `benchmarks/hartmann.py` is configured.

For historical comparability with the committed Camelback reference plot,
`benchmarks/camelback.py` defaults to `--beta-mode legacy`, which passes the
confidence width `sqrt(2 log(t + 1))` explicitly. Use `--beta-mode paper` to
exercise `SafeCtrlBO`'s current paper-style default width based on `B`, `R`,
`delta`, and the information-gain approximation.

The Hartmann script uses the same scalar Hartmann value as both objective and
safety signal, matching the original single-output safe benchmark setup. It
reports simple regret and safety violations.

### Mode-Aware Multi-Task SafeCtrlBO

`src/safectrlbo/multitask_safectrlbo.py` implements the mode-aware extension
for hybrid or contact-rich robot tuning. One rollout chooses one controller
vector `x`, then observes mode-level outputs:

```text
f_free(x), f_transition(x), f_contact(x)
g_free,k(x), g_transition,k(x), g_contact,k(x)
```

The optimizer keeps the hybrid structure:

- utility: `U(x) = sum_m w_m f_m(x)`
- safety: every mode and every safety constraint must satisfy its lower
  confidence bound
- expansion: boundary point with largest mode-wise safety uncertainty
- optimization: certified-safe point with largest weighted utility UCB

`benchmarks/contact_mode_benchmark.py` supports:

- `--surrogate lmc`: physics-guided LMC, one learned mode covariance per kernel
  component
- `--surrogate icm`: simpler separable multi-task kernel
- `--method fused-single-task`: baseline that models weighted utility and
  worst safety margin as scalar outputs
- `--hybrid-discontinuity`: sharper contact-impact safety cliff for stress
  tests

See [docs/multitask_safe_bo.md](docs/multitask_safe_bo.md) for formulation,
data shapes, partial-rollout handling, and simulation-mode guidance.

## Repository Layout

- `src/safectrlbo/`: installable package with the single-task optimizer,
  mode-aware optimizer, GP model helpers, and device/dtype utilities
- `benchmarks/`: maintained Camelback, Hartmann, 12D contact, and suite-runner
  experiment scripts
- `archive/kernel_search/`: historical DARTS-style kernel-search prototype and
  its gantry sample CSV, retained for reference only
- `tests/`: `unittest` coverage for safe-set logic, multi-task shapes, LMC, and
  benchmark helpers
- `docs/multitask_safe_bo.md`: detailed mode-aware multi-task documentation
- `requirements.txt`: dependency pins mirrored by `pyproject.toml`
- `*_simple_regret.png`, `contact_*.png`: committed reference figures

## Environment

The recommended Python baseline is `3.10`.

Known working versions:

- Python 3.10.x
- PyTorch 2.7.x
- GPyTorch 1.15.2
- NumPy 2.2.6
- Matplotlib 3.10.8

Install into a clean virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For `uv`:

```bash
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -e .
```

Notes:

- Scripts support `--device auto`, `--device cpu`, `--device mps`, and
  `--device cuda[:index]`.
- `--device auto` prefers CUDA, then Apple MPS, then CPU.
- On macOS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` if a PyTorch op is not
  implemented on MPS.
- The commands below assume the virtual environment is activated. Without
  activation, use `./.venv/bin/python` instead of `python`.

## Quick Start

Run the unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run a fast Camelback smoke test:

```bash
python -m benchmarks.camelback \
  --num-runs 5 \
  --iterations 20 \
  --num-candidates 4096 \
  --device auto \
  --dtype float64
```

Output:

- console simple-regret summary
- `camelback_simple_regret.png`

Run a fast Hartmann safe-BO smoke test:

```bash
python -m benchmarks.hartmann \
  --num-runs 3 \
  --iterations 20 \
  --num-candidates 2048 \
  --d-effective 1 \
  --rkhs-bound 5.0 \
  --device auto \
  --dtype float64
```

Output:

- console simple-regret summary
- safety-violation report
- `hartmann_simple_regret.png`

Run a 12D mode-aware contact smoke test:

```bash
python -m benchmarks.contact_mode_benchmark \
  --method mode-aware \
  --surrogate lmc \
  --iterations 25 \
  --num-candidates 4096 \
  --num-initial 6 \
  --switch-time 8 \
  --device auto \
  --dtype float64
```

Run the fused single-task baseline:

```bash
python -m benchmarks.contact_mode_benchmark \
  --method fused-single-task \
  --iterations 25 \
  --num-candidates 4096 \
  --num-initial 6 \
  --switch-time 8 \
  --device auto \
  --dtype float64
```

## Reproducing Public Experiment Figures

`benchmarks/run_experiment_suite.py` runs the maintained benchmarks and writes
plots, tables, logs, and machine-readable summaries under `results/`:

```bash
python -m benchmarks.run_experiment_suite \
  --device auto \
  --dtype float64 \
  --hybrid-discontinuity
```

Default suite behavior:

- Camelback: 100 runs x 150 BO steps with `--seed 42` and
  `--camelback-num-candidates 16384`, `--camelback-beta-mode legacy`,
  matching the committed reference result
- Hartmann: 10 runs x 100 BO steps with `--hartmann-d-effective 6`,
  `--hartmann-num-candidates 1024`, `--hartmann-switch-time 5`, and
  `--hartmann-rkhs-bound 2.0`
- Contact benchmark: 10 seeds x 100 BO steps for:
  - fused single-task baseline
  - mode-aware ICM
  - mode-aware LMC

Important output files:

- `results/public_experiments/camelback_simple_regret.png`
- `results/public_experiments/hartmann_simple_regret.png`
- `results/public_experiments/contact_best_safe_utility_improvement.png`
- `results/public_experiments/contact_cumulative_violations.png`
- `results/public_experiments/contact_summary_table.png`
- `results/public_experiments/contact_summary.csv`
- `results/public_experiments/contact_summary.json`
- `results/public_experiments/suite_summary.json`

The repository also includes committed reference figures:

- `camelback_simple_regret.png`
- `hartmann_simple_regret.png`
- `contact_best_safe_utility_improvement.png`
- `contact_cumulative_violations.png`
- `contact_summary_table.png`

To reproduce the committed Camelback reference figure specifically, keep the
larger historical setting:

```bash
python -m benchmarks.camelback \
  --num-runs 100 \
  --iterations 150 \
  --num-candidates 16384 \
  --beta-mode legacy \
  --seed 42 \
  --device auto \
  --dtype float64
```

For contact-only stress tests:

```bash
python -m benchmarks.run_experiment_suite \
  --skip-single-task \
  --num-contact-seeds 10 \
  --contact-iterations 100 \
  --contact-num-candidates 1024 \
  --contact-switch-time 4 \
  --hybrid-discontinuity \
  --impact-threshold 0.45 \
  --impact-penalty 0.30 \
  --device auto \
  --dtype float64
```

This stress setting produced the following CPU result in the current repo over
10 seeds x 100 BO steps:

```text
method              mean improvement      violations      severe
single-task fused   0.1082 +/- 0.0180     6               6
mode-aware ICM      0.0945 +/- 0.0212     0               0
mode-aware LMC      0.1034 +/- 0.0184     2               2
```

The committed Camelback reference (`--beta-mode legacy`) produced:

```text
100 runs x 150 BO steps
step 100 mean regret     8.19e-4
step 100 median regret   4.80e-4
final mean regret        5.98e-4
final median regret      2.72e-4
```

The current Hartmann setting (`B=2.0`, `switch_time=5`) produced:

```text
10 runs x 100 BO steps
final mean regret    0.2009
final median regret  0.2065
safety violations    0 / 1000
```

## Archived Kernel Search

`archive/kernel_search/selectKernel.py` is retained as an archived DARTS-style
kernel-search prototype for the gantry CSV data. It is not required for the
current Camelback, Hartmann, or 12D contact experiments.

The archived command is:

```bash
python archive/kernel_search/selectKernel.py \
  --data archive/kernel_search/gantry_data1.csv \
  --target perf \
  --device auto \
  --dtype float64
```

The old GP-initialization sanity check is archived with the same prototype:

```bash
python archive/kernel_search/gp_initialization.py --device auto --dtype float64
```

The legacy CSV header is:

```text
px1,ix1,dx1,px2,ix2,dx2,py,iy,dy,perf,safe
```

The loader expects the first 9 columns to be input features and `perf` or
`safe` as the target column. The sample file has 30 data rows plus one header.

## Citation

If you find this repo useful, you can consider citing our work:

```bibtex
@ARTICLE{safectrlbo-ral2025,
  author={Wang, Hongxuan and Li, Xiaocong and Zheng, Lihao and Bhaumik, Adrish and Vadakkepat, Prahlad},
  journal={IEEE Robotics and Automation Letters},
  title={Safe Bayesian Optimization for Complex Control Systems via Additive Gaussian Processes},
  year={2025},
  volume={10},
  number={11},
  pages={11538-11545},
  doi={10.1109/LRA.2025.3612756}
}
```
