# Mode-Aware Multi-Task Safe BO

This document describes the multi-task extension in this repository. The
implementation is intended for contact-rich controller tuning where one robot
rollout with one controller parameter vector `x` can produce measurements from
several physical modes, for example `free`, `transition`, and `contact`.

The current implementation is best described as:

```text
mode-aware multi-task safe BO with joint mode observations
```

It is not the cost-aware MTBO setting where the acquisition chooses both `x`
and a separately queryable task `m`. In the robotics setting here, one rollout
chooses only `x`; the logged trajectory may then provide multiple mode-level
performance and safety outputs.

## Formulation

For modes `m in M` and safety constraints `k = 1..K`, the optimizer models:

```text
f_m(x)       mode-level performance
g_m,k(x)     mode-level safety margin
```

The total utility is combined only at the decision layer:

```text
U(x) = sum_m w_m f_m(x)
```

Safety is the intersection of all mode-wise confidence-safe sets:

```text
S_t = { x : LCB_t[g_m,k(x)] >= h_m,k for every mode m and constraint k }
```

This avoids fitting a GP to `min_m,k g_m,k(x)`. The `min` target is often
non-smooth near contact transitions even when each physical mode is smoother
on its own.

## Data Shapes

`MultiTaskSafeCtrlBO` expects complete joint observations by default:

```python
init_X.shape       == (n, d)
init_Y_perf.shape  == (n, num_modes)
init_Y_safe.shape  == (n, num_modes, num_safety_constraints)
```

For the default contact benchmark:

```text
num_modes = 3
modes = free, transition, contact

num_safety_constraints = 2
constraints = force_margin, stability_margin
```

Each trial observes:

```text
Y_perf[i] =
  [f_free(x_i), f_transition(x_i), f_contact(x_i)]

Y_safe[i] =
  [
    [g_free,force(x_i),       g_free,stability(x_i)],
    [g_transition,force(x_i), g_transition,stability(x_i)],
    [g_contact,force(x_i),    g_contact,stability(x_i)]
  ]
```

Positive safety margins mean safe by default; thresholds can be scalar, one
value per safety constraint, or a full `(num_modes, num_constraints)` matrix.

## Surrogate Models

The implementation uses:

```text
one multi-task GP for f(x, m)
one multi-task GP per safety constraint g_k(x, m)
```

Two mode-aware surrogate kernels are supported.

### Physics-Guided LMC

The default kernel is physics-guided LMC:

```text
k((x,m), (x',m')) = sum_q k_q(x, x') B_q,m,m'
```

Each latent kernel `k_q` has its own learned mode covariance `B_q`. In the 12D
contact benchmark, the `k_q` components are the physics groups:

- free-space tracking and speed gains
- impedance stiffness/damping block
- force/impedance shaping block
- speed-stiffness contact impact interaction
- contact compliance interaction

This lets an impact-related component learn a different free/transition/contact
correlation than a tracking-related component. It is the default because the
current benchmark results make it the best performance/safety tradeoff among
the mode-aware models: more utility than ICM in long stress runs, and far fewer
unsafe decisions than the fused single-task baseline.

For safety, the first LMC implementation keeps the data kernels frozen and
trains task covariance and likelihood noise. This limits overfitting in the
small-data regime while still allowing different physical factors to learn
different mode correlations.

### ICM

ICM remains available through `--surrogate icm`:

```text
k((x,m), (x',m')) = k_x(x, x') B_m,m'
```

where `k_x` is the provided controller-parameter kernel and `B` is one learned
mode covariance matrix from GPyTorch `MultitaskKernel`.

ICM is a useful fallback and ablation. It is simpler and can be more stable
when data are extremely scarce, but it assumes all modes share the same
input-space smoothness and active kernel structure.

In the current contact stress test, ICM is the most conservative option:

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

Use LMC as the default balanced surrogate. Use ICM when the experiment is
strictly safety-first and accepting lower utility is preferable to any unsafe
trial.

## Confidence Widths

The default confidence width follows the paper-style form:

```text
beta_t = B + R sqrt(2(gamma_{t-1} + 1 + log(1 / delta)))
```

`MultiTaskSafeCtrlBO` now separates objective and safety confidence widths:

```python
beta_f_fn(t)  # for utility UCB
beta_g_fn(t)  # for safety LCB
```

When no custom `beta_fn` is supplied, safety uses a simple mode/constraint
union correction:

```text
delta_g = delta / (num_modes * num_safety_constraints)
```

This makes safety LCBs more conservative than objective UCBs because the safe
set requires every mode and every constraint to be calibrated at the same time.

For backward compatibility, passing `beta_fn` still overrides both `beta_f_fn`
and `beta_g_fn`.

## Acquisition Logic

The algorithm has two phases:

1. Expansion phase: search the safe boundary.
2. Optimization phase: optimize weighted utility inside the safe set.

`switch_time` now means the number of BO trials spent in expansion after the
initial safe data. Initial safe points no longer consume `switch_time`.

Expansion uses the boundary point with largest safety uncertainty by default:

```text
argmax_x in boundary max_m,k sigma[g_m,k(x)]
```

Optimization uses utility UCB:

```text
argmax_x in safe_set mu_U(x) + beta_f sigma_U(x)
```

The utility mean is:

```text
mu_U(x) = sum_m w_m mu[f_m(x)]
```

The utility uncertainty uses a conservative scalable bound:

```text
sigma_U(x) <= sum_m |w_m| sigma[f_m(x)]
```

This avoids the optimistic independence approximation without materializing
dense posterior task covariance blocks for every Sobol candidate.

## Partial Rollouts and Aborts

Real robot trials may abort before every mode is observed. For example:

```text
free observed
transition unsafe
contact not executed
```

The class provides `observe_partial(...)` for this first practical case. It
accepts the observed modes and fills missing modes conservatively:

```python
bo.observe_partial(
    x_new=x,
    y_perf_new=[[f_free, f_transition]],
    y_safe_new=[[
        [g_free_force, g_free_stability],
        [g_trans_force, g_trans_stability],
    ]],
    observed_modes=("free", "transition"),
    missing_reason="unsafe_abort",
    missing_perf_value=-2.0,
    missing_safety_value=-1.0,
)
```

The fill strategy keeps the matrix-target MTGP implementation simple:

- missing performance receives a low utility value
- missing safety receives a value below threshold
- the aborted trial cannot be treated as empirically safe

This is conservative and useful for unsafe robot aborts. It should not be used
for every missing-data case:

- abort due to unsafe transition: conservative fill is appropriate
- no contact because the task never reached contact: model contact occurrence
  or use masked/ragged observations
- sensor or logging failure: do not treat the missing mode as unsafe

`observe_partial(...)` therefore rejects explicit non-abort reasons such as
`missing_reason="no_contact"` until a masked/ragged data path is implemented.
A more complete future version should support ragged observations directly,
either through flattened `(x, mode, y)` task-index data or independent per-mode
GPs for safety.

## Contact Benchmark

`contact_mode_benchmark.py` defines a 12D controller-tuning problem with:

- speed/tracking gains
- stiffness/damping terms
- force gain and impedance terms
- `free`, `transition`, and `contact` performance outputs
- force and stability safety margins per mode

Run a smooth smoke test:

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

Run the sharper contact-transition benchmark:

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

The hybrid flag adds an impact index:

```text
impact = speed * stiffness + 0.8 * impedance - 0.7 * damping
```

When the impact index crosses the threshold, transition force and stability
margins receive a sharp penalty. This makes the benchmark closer to a
contact-rich hybrid system where transition safety is less smooth than
free-space tracking.

The benchmark summary reports:

- `best_safe_utility`
- `estimated_best_feasible_utility`
- `safety_violations`
- `violation_rate`
- `certified_decision_count`
- `certified_false_safe_count`
- `certified_false_safe_rate`
- `severe_violations`
- `unsafe_worst_mode_constraint_counts`

For contact-rich experiments, false-safe rate and severe violations are more
important than simple regret alone.

Initial safe points are filtered using noise-free safety margins, even when the
stored observations include measurement noise. This avoids accepting a truly
unsafe initial point just because observation noise made its safety margin
positive.

The same script can run the fused single-task baseline:

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

The fused baseline trains one objective GP on:

```text
f(x) = sum_m w_m f_m(x)
```

and one safety GP on:

```text
g(x) = min_m,k g_m,k(x)
```

It is useful as a direct comparison against the mode-aware method, but it is
expected to over-smooth contact-transition safety boundaries in harder hybrid
settings.

For an ICM vs LMC ablation, keep all other flags fixed and switch only:

```bash
--surrogate lmc
```

versus:

```bash
--surrogate icm
```

For a longer stress comparison matching the table above, use:

```bash
python contact_mode_benchmark.py \
  --method mode-aware \
  --surrogate lmc \
  --iterations 100 \
  --num-candidates 1024 \
  --num-initial 6 \
  --switch-time 4 \
  --hybrid-discontinuity \
  --impact-threshold 0.45 \
  --impact-penalty 0.30 \
  --train-hypers-every 10 \
  --training-iter 2 \
  --device auto \
  --dtype float64
```

## Defining Modes in Simulation

If the simulator exposes contact state, use event-based segmentation:

```text
free:        no contact before first contact event
transition:  short window around contact onset or loss
contact:     sustained contact after the event
```

If contact timing is unknown before execution, you do not need to predict it in
advance. Choose `x`, run the trajectory, then segment the logged trajectory
afterward using contact force, distance-to-surface, contact flags, velocity
jumps, or controller events.

When a rollout contains multiple free/contact episodes, aggregate mode-level
metrics over all segments:

```text
f_free(x)        = average or worst free tracking score
g_transition(x)  = minimum safety margin over all contact onsets
g_contact(x)     = minimum sustained-contact safety margin
```

For safety, worst-case aggregation is usually preferable:

```text
g_m,k(x) = min over segments of margin_m,k(segment)
```

This keeps the BO input as a single controller vector `x` while still preserving
the hybrid structure in the outputs.

## Current Limitations

The current implementation deliberately keeps the first multi-task version
small and testable. Important future extensions are:

- exact `w^T Sigma w` utility variance refinement for top-K candidates
- output normalization for real robot metrics with different units
- independent per-mode safety GP baseline for missing data
- shared + mode-specific residual kernels for negative-transfer control if LMC
  shows overfitting or negative transfer on larger robot logs
- gated or changepoint kernels for contact-onset discontinuities
- low-level robot safety filters such as force ramps, passivity checks, CBFs,
  torque saturation, and emergency stop logic

The BO-level safe set is a statistical decision rule. It should not replace
controller-level safety checks on real hardware.
