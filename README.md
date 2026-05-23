# SafeCtrlBO

Experimental GPyTorch code for safe Bayesian optimization with current workflows:

- `camelback.py`: run repeated Bayesian optimization on a 2D camelback benchmark using `SafeCtrlBO`.
- `hartmann.py`: run a safe 6D Hartmann benchmark with additive kernels and safety-violation reporting.
- `contact_mode_benchmark.py`: run a 12D mode-aware safe BO benchmark with free, transition, and contact modes.
- `selectKernel.py`: legacy DARTS-style kernel search from gantry controller data.

The single-task BO loop is implemented in `safectrlbo.py`, the mode-aware
multi-task BO loop is implemented in `multitask_safectrlbo.py`, GP model
helpers live in `model.py`, and reusable runtime helpers live in
`device_utils.py`.

## What This Repo Does

The main idea is:

1. Learn or select a GP kernel structure from data.
2. Freeze that kernel.
3. Use it inside a `SafeCtrlBO` optimizer that can operate in either:
   - safe mode, when a safety signal and threshold are provided
   - unconstrained mode, when they are omitted

There are currently two separate tracks in the repo:

- Gantry kernel search legacy path
  - `selectKernel.py` loads 9D controller data from `gantry_data1.csv`
  - it performs a DARTS-style bilevel search over kernel structure
  - it prints a copy-pasteable frozen kernel snippet at the end
  - this path is retained for reference; current kernel selection work has moved to the newer SPARK workflow outside this script

- Camelback benchmark
  - `camelback.py` runs many BO trials on a 2D synthetic objective
  - it uses `SafeCtrlBO` in unconstrained mode
  - it writes `camelback_simple_regret.png`

- Hartmann safe benchmark
  - `hartmann.py` runs safe BO on a 6D synthetic objective
  - it reports simple regret and safety violations
  - it exposes `--rkhs-bound`, `--noise-bound`, `--delta`, and `--expansion-uncertainty` for confidence-width and expansion ablations

- Contact-mode benchmark
  - `contact_mode_benchmark.py` runs mode-aware safe BO on a 12D synthetic controller problem
  - each trial observes free, transition, and contact mode metrics
  - the safe set is the intersection of all mode-wise force and stability margins
  - the objective is a weighted utility over mode-level performance values
  - default `--surrogate lmc` uses physics-guided LMC over the additive kernel groups as a performance/safety tradeoff
  - optional `--surrogate icm` runs the simpler separable multi-task kernel and is currently the most conservative stress-test option
  - optional `--hybrid-discontinuity` adds a sharp contact-impact safety cliff
  - optional `--method fused-single-task` runs the old fused baseline `f=sum w_m f_m`, `g=min g_m,k`

## Repository Layout

- `safectrlbo.py`: main optimization loop, candidate generation, safe-set logic, and observation updates
- `multitask_safectrlbo.py`: mode-aware multi-task safe BO for hybrid contact-rich settings
- `model.py`: exact GP and multi-task GP wrappers plus fitting utilities
- `selectKernel.py`: legacy kernel-structure search on CSV data
- `kernels.py`: frozen additive kernel constructor currently used by `gp_initialization.py`
- `gp_initialization.py`: minimal sanity check that instantiates performance and safety GPs
- `camelback.py`: 2D benchmark script for repeated BO runs and regret plotting
- `device_utils.py`: device and dtype helpers
- `gantry_data1.csv`: sample gantry dataset with 9 inputs plus `perf` and `safe`

## Environment

The original Linux/CUDA environment for this repo used:

- Python 3.10.19
- NumPy 2.2.6
- PyTorch 2.10.0
- GPyTorch 1.15.2
- Matplotlib 3.10.8

For a fresh macOS or Linux setup, use Python `3.10` plus the
cross-platform `requirements.txt` below, and choose the appropriate PyTorch
wheel for your platform when you need GPU support.

## Installation

Create a clean virtual environment and install the core dependencies from
`requirements.txt`.

### Recommended: `uv` on macOS or Linux

```bash
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

### Standard library `venv`

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:

- The recommended Python baseline for this repo is `3.10`.
- On Apple Silicon Macs, PyTorch uses the `mps` device for GPU acceleration. `--device auto` now prefers `cuda`, then `mps`, then `cpu`.
- If you want NVIDIA CUDA support on Linux, install the matching PyTorch wheel from the official selector first, then install `requirements.txt`. For example, CUDA 12.8:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.8"
uv pip install -r requirements.txt
```

- On macOS, if a specific op is not implemented on MPS yet, PyTorch can fall back to CPU when `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before launching Python:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

- The scripts support `--device auto`, `--device cpu`, `--device mps`, and `--device cuda[:index]`.
- `camelback.py` configures a non-interactive Matplotlib backend, so it works in headless environments.
- The commands below assume you activated a virtual environment. If you prefer to use the committed environment in this repo, replace `python` with `./env/bin/python`.

## Quick Start

### 1. Sanity Check GP Initialization

This is the fastest way to confirm the basic GP code path is working:

```bash
python gp_initialization.py --device auto --dtype float64
```

Expected output:

```text
Initializing GPs with device=..., dtype=float64
Initialized performance and safety GPs.
```

On Apple Silicon, you can also verify MPS directly:

```bash
python -c "import torch; print('mps built:', torch.backends.mps.is_built()); print('mps available:', torch.backends.mps.is_available())"
python gp_initialization.py --device mps --dtype float64
```

### 2. Run the Camelback Benchmark

```bash
python camelback.py --device auto --dtype float64
```

Useful flags:

- `--num-runs`: number of repeated BO runs, default `100`
- `--iterations`: BO steps per run, default `150`
- `--num-candidates`: Sobol candidates evaluated per step, default `16384`
- `--seed`: base seed for reproducibility
- `--success-threshold`: threshold used when reporting success rate

Output:

- console summary statistics over simple regret
- `camelback_simple_regret.png`

Example for a faster smoke test:

```bash
python camelback.py --num-runs 5 --iterations 20 --num-candidates 4096 --device auto --dtype float64
```

### 3. Run the Hartmann Safe Benchmark

```bash
python hartmann.py --device auto --dtype float64
```

Useful flags:

- `--d-effective`: maximum additive interaction order, default `6`
- `--safety-threshold`: safety threshold, default `0.3`
- `--rkhs-bound`: RKHS bound `B` used in the paper-style confidence width
- `--noise-bound`: sub-Gaussian noise bound `R`; defaults to the likelihood-noise standard deviation
- `--delta`: failure probability used in the confidence width
- `--expansion-uncertainty {safety,objective}`: use Algorithm 1 safety uncertainty or the earlier objective-uncertainty ablation

Example for a faster conservative smoke test:

```bash
python hartmann.py \
  --num-runs 3 \
  --iterations 20 \
  --num-candidates 2048 \
  --d-effective 1 \
  --rkhs-bound 5.0 \
  --device auto \
  --dtype float64
```

### 4. Run Kernel Search on the Gantry Data

This script is a legacy DARTS-style path. It is still useful as a reference,
but the newer SPARK kernel-selection workflow is not implemented in
`selectKernel.py`.

```bash
python selectKernel.py --data gantry_data1.csv --target perf --device auto --dtype float64
```

Useful flags:

- `--target {perf,safe}`: choose which CSV target to model
- `--max_order`: maximum interaction order in the kernel search space
- `--outer_steps`: outer bilevel optimization steps
- `--inner_steps`: inner hyperparameter optimization steps
- `--topk`: number of components to include in the exported snippet

Example:

```bash
python selectKernel.py \
  --data gantry_data1.csv \
  --target perf \
  --max_order 2 \
  --outer_steps 100 \
  --inner_steps 2 \
  --device auto \
  --dtype float64
```

At the end of a run, the script prints:

- best validation objective
- top mixture components
- learned shared lengthscales
- GP likelihood noise
- a copy-pasteable kernel snippet for reuse

## Data Format

`gantry_data1.csv` currently uses a header of:

```text
px1,ix1,dx1,px2,ix2,dx2,py,iy,dy,perf,safe
```

The loader in `selectKernel.py` expects:

- the first 9 columns to be input features
- `perf` or `safe` as the target column when a header is present
- at least enough valid rows to survive cleaning and train/validation splitting

The current sample file has 30 data rows plus a header row.

## How SafeCtrlBO Works

`SafeCtrlBO` is initialized with:

- observed inputs `init_X`
- performance observations `init_Y_perf`
- optional safety observations `init_Y_safe`; use shape `(n, m)` for `m` safety constraints
- box constraints `bounds`
- a frozen `base_kernel`
- optional paper-style confidence parameters `rkhs_bound`, `noise_bound`, `delta`, and `information_gain_fn`

Behavior depends on whether safety data is supplied:

- Safe mode:
  - uses one separate GP per safety constraint
  - certifies safe candidates only when every safety lower confidence bound satisfies its threshold
  - accepts either a scalar `safety_threshold` or one threshold per safety constraint
  - expands or optimizes only within the safe set
  - uses the relaxed boundary set from the paper; if it is empty, it falls back to safe candidates with the smallest lower-confidence safety margin
  - by default expands with `max_i sigma_g_i` on the boundary, matching Algorithm 1; pass `expansion_uncertainty="objective"` to reproduce the earlier objective-uncertainty ablation
  - uses the paper-style width `beta_t = B + R sqrt(2(gamma_{t-1} + 1 + log(1/delta)))` by default; calibrate `B`, `R`, and `gamma` for practical safety
  - falls back to certified-safe observed points when needed

- Unconstrained mode:
  - if `init_Y_safe=None` and `safety_threshold=None`, all candidates are treated as safe
  - this is how `camelback.py` currently runs

## How MultiTaskSafeCtrlBO Works

`MultiTaskSafeCtrlBO` is for contact-rich or hybrid settings where one trial
with controller parameters `x` produces mode-level measurements:

```text
f_free(x), f_transition(x), f_contact(x)
g_free,k(x), g_transition,k(x), g_contact,k(x)
```

The implementation uses one GPyTorch multi-task GP for mode-level performance
and one multi-task GP per safety constraint. The input kernel models controller
parameters, while `MultitaskKernel` learns a mode covariance over free,
transition, and contact outputs. The default `--surrogate lmc` uses one mode
covariance per physics-guided kernel component; `--surrogate icm` is the simpler
alternative with one shared input kernel and one mode covariance.

In the current 12D stress benchmark, LMC is the default because it gives a
better utility/safety tradeoff than ICM, while ICM remains the safer
fallback when zero unsafe trials is the top priority.

```text
Stress setting:
  10 seeds x 100 BO steps
  hybrid discontinuity enabled
  impact_threshold=0.45, impact_penalty=0.30

method              mean improvement      violations      severe
single-task fused   0.108218 +/- 0.018042  6               6
multi-task ICM      0.097131 +/- 0.018779  0               0
multi-task LMC      0.098880 +/- 0.020567  1               1
```

Use `--surrogate lmc` for the balanced default and `--surrogate icm` for a
more conservative safe-robotics run.

The acquisition keeps the hybrid structure instead of fitting a GP to a fused
minimum:

- utility: `U(x) = sum_m w_m f_m(x)`
- safe set: every mode and every safety constraint must satisfy its LCB
- expansion: boundary point with largest mode-wise safety uncertainty
- optimization: safe candidate with largest weighted utility UCB
- `switch_time`: number of BO trials spent in expansion after the initial safe data
- safety beta: default confidence width applies a `num_modes * num_constraints` union correction
- unsafe aborts: `observe_partial(..., missing_reason="unsafe_abort")` conservatively fills unobserved modes below threshold

The utility UCB uses a conservative weighted uncertainty bound,
`sum_m |w_m| sigma_m`, rather than assuming posterior independence between
modes.

Run the 12D smoke benchmark with:

```bash
python contact_mode_benchmark.py \
  --method mode-aware \
  --surrogate lmc \
  --iterations 25 \
  --num-candidates 4096 \
  --num-initial 6 \
  --switch-time 8 \
  --device auto \
  --dtype float64
```

Run the sharper hybrid/contact-transition benchmark with online task-covariance
refits:

```bash
python contact_mode_benchmark.py \
  --method mode-aware \
  --surrogate lmc \
  --iterations 25 \
  --num-candidates 4096 \
  --num-initial 6 \
  --switch-time 8 \
  --hybrid-discontinuity \
  --impact-threshold 0.55 \
  --impact-sharpness 80 \
  --impact-penalty 0.20 \
  --train-hypers-every 5 \
  --training-iter 5 \
  --device auto \
  --dtype float64
```

For a safety-first ICM run, keep the same command and change only:

```bash
--surrogate icm
```

The benchmark summary reports `violation_rate`,
`certified_false_safe_rate`, `severe_violations`, and
`unsafe_worst_mode_constraint_counts` in addition to best safe utility. Initial
safe points are filtered with noise-free safety margins before noisy
observations are stored. See `docs/multitask_safe_bo.md` for the full
formulation, data shapes, abort handling, and simulation-mode guidance.

For the fused single-task baseline:

```bash
python contact_mode_benchmark.py \
  --method fused-single-task \
  --iterations 25 \
  --num-candidates 4096 \
  --num-initial 6 \
  --switch-time 8 \
  --hybrid-discontinuity \
  --device auto \
  --dtype float64
```

# Citation

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
