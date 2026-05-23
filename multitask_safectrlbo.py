import math

import torch
import gpytorch
from torch.quasirandom import SobolEngine

from device_utils import resolve_device
from model import build_mode_lmc_gp, build_mode_task_gp, fit_mode_task_gp


class MultiTaskSafeCtrlBO:
    """
    Mode-aware safe BO for hybrid contact-rich manipulation.

    The optimizer selects one controller parameter vector x per trial, while
    modeling mode-level outputs f_m(x) and g_{m,k}(x) for modes such as free,
    transition, and contact. Safety is the intersection over all modes and all
    safety constraints; the objective acquisition is built from a weighted
    mode-level utility.
    """

    def __init__(
        self,
        init_X,
        init_Y_perf,
        init_Y_safe,
        bounds,
        base_kernel,
        safety_threshold,
        mode_names=("free", "transition", "contact"),
        mode_weights=None,
        switch_time=10,
        beta_fn=None,
        beta_f_fn=None,
        beta_g_fn=None,
        tau=0.1,
        device="cpu",
        init_training_iter=0,
        likelihood_noise=1e-4,
        sobol_seed=None,
        safe_retry_radius=0.05,
        rkhs_bound=1.0,
        noise_bound=None,
        delta=0.05,
        information_gain_fn=None,
        task_rank=1,
        expansion_uncertainty="safety",
        missing_perf_value=None,
        missing_safety_value=None,
        multitask_kernel="lmc",
    ):
        self.device = resolve_device(device)
        self.bounds = bounds.to(self.device)
        self.mode_names = tuple(mode_names)
        self.num_modes = len(self.mode_names)
        self.task_rank = int(task_rank)
        if multitask_kernel not in {"icm", "lmc"}:
            raise ValueError("multitask_kernel must be 'icm' or 'lmc'.")
        self.multitask_kernel = multitask_kernel

        if expansion_uncertainty not in {"safety", "utility"}:
            raise ValueError("expansion_uncertainty must be 'safety' or 'utility'.")
        self.expansion_uncertainty = expansion_uncertainty

        self.switch_time = switch_time
        self.tau = tau
        self.likelihood_noise = likelihood_noise
        self.safe_retry_radius = safe_retry_radius
        self.rkhs_bound = float(rkhs_bound)
        if noise_bound is None:
            self.noise_bound = math.sqrt(float(likelihood_noise)) if likelihood_noise is not None else 1.0
        else:
            self.noise_bound = float(noise_bound)
        if not 0.0 < float(delta) < 1.0:
            raise ValueError("delta must be in (0, 1).")
        self.delta = float(delta)
        self.information_gain_fn = information_gain_fn or self._default_information_gain
        self._legacy_beta_fn = beta_fn
        self._provided_beta_f_fn = beta_f_fn
        self._provided_beta_g_fn = beta_g_fn
        self.missing_perf_value = missing_perf_value
        self.missing_safety_value = missing_safety_value

        self._sobol_engine = SobolEngine(
            dimension=self.bounds.shape[1],
            scramble=True,
            seed=sobol_seed,
        )

        self.X = init_X.to(self.device)
        self.Yf = self._format_mode_observations(init_Y_perf, expected_rows=self.X.shape[0])
        self.Yg = self._format_safety_observations(init_Y_safe, expected_rows=self.X.shape[0])
        self.num_safety_constraints = self.Yg.shape[2]
        self.safety_thresholds = self._format_safety_thresholds(safety_threshold)
        self.mode_weights = self._format_mode_weights(mode_weights)
        self.n_initial = self.X.shape[0]
        self.n_iter = self.X.shape[0]
        self.bo_steps = 0
        self._configure_beta_functions()

        self.rebuild_models(base_kernel, training_iter=init_training_iter)

    def _default_information_gain(self, t):
        t_value = max(int(t), 0)
        return torch.log(torch.tensor(float(t_value + 1.0), device=self.device))

    def _default_beta_fn(self, n, delta=None):
        beta_delta = self.delta if delta is None else float(delta)
        if not 0.0 < beta_delta < 1.0:
            raise ValueError("beta delta must be in (0, 1).")
        t_minus_one = max(int(n) - 1, 0)
        gamma = torch.as_tensor(
            self.information_gain_fn(t_minus_one),
            dtype=self.X.dtype,
            device=self.device,
        )
        confidence = 2.0 * (gamma + 1.0 + math.log(1.0 / beta_delta))
        return self.rkhs_bound + self.noise_bound * torch.sqrt(confidence)

    def _configure_beta_functions(self):
        if self._legacy_beta_fn is not None:
            self.beta_f_fn = self._legacy_beta_fn
            self.beta_g_fn = self._legacy_beta_fn
            return

        safety_delta = self.delta / float(self.num_modes * self.num_safety_constraints)
        self.beta_f_fn = self._provided_beta_f_fn or (lambda n: self._default_beta_fn(n, self.delta))
        self.beta_g_fn = self._provided_beta_g_fn or (lambda n: self._default_beta_fn(n, safety_delta))

    def _format_mode_observations(self, values, expected_rows):
        y = torch.as_tensor(values, dtype=self.X.dtype, device=self.device)
        if y.dim() == 1 and expected_rows == 1 and y.numel() == self.num_modes:
            y = y.view(1, self.num_modes)
        elif y.dim() != 2:
            raise ValueError(
                f"Mode observations must have shape (n, {self.num_modes}) "
                f"or ({self.num_modes},) for one row."
            )
        if y.shape != (expected_rows, self.num_modes):
            raise ValueError(
                f"Expected mode observations with shape ({expected_rows}, {self.num_modes}), "
                f"got {tuple(y.shape)}."
            )
        return y

    def _format_safety_observations(self, values, expected_rows):
        y = torch.as_tensor(values, dtype=self.X.dtype, device=self.device)
        if y.dim() == 1 and expected_rows == 1 and y.numel() == self.num_modes:
            y = y.view(1, self.num_modes, 1)
        elif y.dim() == 2:
            if expected_rows == 1 and y.shape[0] == self.num_modes:
                y = y.view(1, self.num_modes, y.shape[1])
            elif y.shape == (expected_rows, self.num_modes):
                y = y.unsqueeze(-1)
            else:
                raise ValueError(
                    "2D safety observations must be (num_modes, k) for one row "
                    "or (n, num_modes) for one safety constraint."
                )
        elif y.dim() != 3:
            raise ValueError("Safety observations must be 1D, 2D, or 3D.")

        if y.shape[0] != expected_rows or y.shape[1] != self.num_modes:
            raise ValueError(
                f"Expected safety observations with first dimensions "
                f"({expected_rows}, {self.num_modes}), got {tuple(y.shape)}."
            )
        return y

    def _format_safety_thresholds(self, safety_threshold):
        thresholds = torch.as_tensor(
            safety_threshold,
            dtype=self.X.dtype,
            device=self.device,
        )
        if thresholds.dim() == 0 or thresholds.numel() == 1:
            thresholds = thresholds.reshape(1, 1).expand(
                self.num_modes,
                self.num_safety_constraints,
            )
        elif thresholds.dim() == 1:
            if thresholds.numel() == self.num_safety_constraints:
                thresholds = thresholds.view(1, -1).expand(self.num_modes, -1)
            elif thresholds.numel() == self.num_modes and self.num_safety_constraints == 1:
                thresholds = thresholds.view(-1, 1)
            else:
                raise ValueError(
                    "1D safety_threshold must have length num_constraints, or "
                    "num_modes when there is one safety constraint."
                )
        elif thresholds.dim() == 2:
            if thresholds.shape != (self.num_modes, self.num_safety_constraints):
                raise ValueError(
                    f"Expected safety_threshold shape "
                    f"({self.num_modes}, {self.num_safety_constraints}), got {tuple(thresholds.shape)}."
                )
        else:
            raise ValueError("safety_threshold must be scalar, 1D, or 2D.")
        return thresholds

    def _format_mode_weights(self, mode_weights):
        if mode_weights is None:
            weights = torch.ones(self.num_modes, dtype=self.X.dtype, device=self.device)
            weights = weights / weights.sum()
            return weights

        weights = torch.as_tensor(mode_weights, dtype=self.X.dtype, device=self.device).reshape(-1)
        if weights.numel() != self.num_modes:
            raise ValueError(f"Expected {self.num_modes} mode weights, got {weights.numel()}.")
        if torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("mode_weights must be non-negative and sum to a positive value.")
        return weights / weights.sum()

    def rebuild_models(self, base_kernel, training_iter=0):
        builder = build_mode_lmc_gp if self.multitask_kernel == "lmc" else build_mode_task_gp
        self.model_f, self.lik_f, self.mll_f = builder(
            self.X,
            self.Yf,
            base_kernel,
            num_modes=self.num_modes,
            noise=self.likelihood_noise,
            task_rank=self.task_rank,
        )

        self.models_g = []
        self.liks_g = []
        self.mlls_g = []
        for constraint_idx in range(self.num_safety_constraints):
            model_g, lik_g, mll_g = builder(
                self.X,
                self.Yg[:, :, constraint_idx],
                base_kernel,
                num_modes=self.num_modes,
                noise=self.likelihood_noise,
                task_rank=self.task_rank,
            )
            self.models_g.append(model_g)
            self.liks_g.append(lik_g)
            self.mlls_g.append(mll_g)

        if training_iter is not None and training_iter > 0:
            self._fit_models(training_iter=training_iter)

    def _fit_models(self, training_iter=0):
        fit_mode_task_gp(
            self.model_f,
            self.lik_f,
            self.mll_f,
            training_iter=training_iter,
            train_data_kernel=False,
            train_task_covar=True,
            train_mean=False,
            train_noise=True,
        )
        for model_g, lik_g, mll_g in zip(self.models_g, self.liks_g, self.mlls_g):
            fit_mode_task_gp(
                model_g,
                lik_g,
                mll_g,
                training_iter=training_iter,
                train_data_kernel=False,
                train_task_covar=True,
                train_mean=False,
                train_noise=True,
            )

    @torch.no_grad()
    def posterior_mode_mean_std(self, model, likelihood, Xtest):
        model.eval()
        if likelihood is not None:
            likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            pred = model(Xtest)
        mean = pred.mean
        std = pred.variance.clamp_min(0.0).sqrt()
        return mean, std

    def posterior_perf_by_mode(self, Xtest):
        return self.posterior_mode_mean_std(self.model_f, self.lik_f, Xtest)

    def posterior_safety_by_mode(self, Xtest):
        means = []
        stds = []
        for model_g, lik_g in zip(self.models_g, self.liks_g):
            mean_g, std_g = self.posterior_mode_mean_std(model_g, lik_g, Xtest)
            means.append(mean_g)
            stds.append(std_g)
        return torch.stack(means, dim=-1), torch.stack(stds, dim=-1)

    def _beta_width(self, beta, dtype):
        width = torch.as_tensor(beta, dtype=dtype, device=self.device)
        if torch.any(width < 0):
            raise ValueError("beta confidence width must be non-negative.")
        return width

    def _utility_mean_std(self, mean_f, std_f):
        # Conservative scalable bound on the weighted utility uncertainty:
        # std(sum_m w_m f_m) <= sum_m |w_m| std(f_m). This avoids the optimistic
        # independence approximation without materializing dense task blocks.
        weights = self.mode_weights.to(dtype=mean_f.dtype).view(1, -1)
        utility_mean = (mean_f * weights).sum(dim=-1)
        utility_std = (std_f * weights.abs()).sum(dim=-1)
        return utility_mean, utility_std

    def _observed_safe_points(self, beta_g):
        beta_width = self._beta_width(beta_g, self.X.dtype)
        mu_g_obs, std_g_obs = self.posterior_safety_by_mode(self.X)
        l_g_obs = mu_g_obs - beta_width * std_g_obs
        safe_obs_mask = torch.all(
            l_g_obs >= self.safety_thresholds.view(1, self.num_modes, self.num_safety_constraints),
            dim=(1, 2),
        )
        return self.X[safe_obs_mask]

    def _empirically_safe_observed_points(self):
        safe_obs_mask = torch.all(
            self.Yg >= self.safety_thresholds.view(1, self.num_modes, self.num_safety_constraints),
            dim=(1, 2),
        )
        return self.X[safe_obs_mask]

    def _local_safe_retry_candidates(self, safe_points, num_candidates):
        if safe_points.numel() == 0:
            return safe_points

        safe_points = safe_points.to(device=self.device, dtype=self.bounds.dtype)
        num_safe = safe_points.shape[0]
        num_local = max(num_candidates - num_safe, 0)
        if num_local == 0:
            return safe_points

        sample_ids = torch.randint(num_safe, (num_local,), device=self.device)
        centers = safe_points[sample_ids]
        span = (self.bounds[1] - self.bounds[0]).unsqueeze(0)
        perturb_scale = self.safe_retry_radius * span
        X_local = centers + torch.randn_like(centers) * perturb_scale
        lb = self.bounds[0].unsqueeze(0)
        ub = self.bounds[1].unsqueeze(0)
        X_local = torch.maximum(torch.minimum(X_local, ub), lb)
        return torch.cat([safe_points, X_local], dim=0)

    def _best_empirically_safe_observed_point(self):
        safe_obs_mask = torch.all(
            self.Yg >= self.safety_thresholds.view(1, self.num_modes, self.num_safety_constraints),
            dim=(1, 2),
        )
        if not torch.any(safe_obs_mask):
            raise RuntimeError("MultiTaskSafeCtrlBO has no empirically safe observed point.")
        utilities = (self.Yf[safe_obs_mask] * self.mode_weights.view(1, -1)).sum(dim=-1)
        return self.X[safe_obs_mask][torch.argmax(utilities)]

    def _best_certified_safe_observed_point(self, beta_f, beta_g):
        safe_points = self._observed_safe_points(beta_g)
        if safe_points.shape[0] == 0:
            raise RuntimeError("MultiTaskSafeCtrlBO could not certify any observed point as safe.")
        mean_f, std_f = self.posterior_perf_by_mode(safe_points)
        utility_mean, utility_std = self._utility_mean_std(mean_f, std_f)
        beta_width = self._beta_width(beta_f, safe_points.dtype)
        utility_u = utility_mean + beta_width * utility_std
        return safe_points[torch.argmax(utility_u)]

    def _get_sets(self, X_cand, beta=None, beta_f=None, beta_g=None):
        X_cand = X_cand.to(device=self.device, dtype=self.X.dtype)
        if beta is not None:
            beta_f = beta if beta_f is None else beta_f
            beta_g = beta if beta_g is None else beta_g
        if beta_f is None:
            beta_f = self.beta_f_fn(self.n_iter)
        if beta_g is None:
            beta_g = self.beta_g_fn(self.n_iter)
        beta_f_width = self._beta_width(beta_f, X_cand.dtype)
        beta_g_width = self._beta_width(beta_g, X_cand.dtype)

        mean_f, std_f = self.posterior_perf_by_mode(X_cand)
        utility_mean, utility_std = self._utility_mean_std(mean_f, std_f)
        utility_u = utility_mean + beta_f_width * utility_std

        mu_g, std_g = self.posterior_safety_by_mode(X_cand)
        l_g = mu_g - beta_g_width * std_g
        thresholds = self.safety_thresholds.view(1, self.num_modes, self.num_safety_constraints)
        per_constraint_margin = l_g - thresholds
        safety_margin = per_constraint_margin.amin(dim=(1, 2))
        safe_mask = torch.all(per_constraint_margin >= 0.0, dim=(1, 2))

        S = X_cand[safe_mask]
        tau = torch.as_tensor(self.tau, dtype=X_cand.dtype, device=self.device)
        boundary_mask = safe_mask & (safety_margin <= tau)
        B = X_cand[boundary_mask]
        if B.shape[0] == 0 and S.shape[0] > 0:
            safe_margins = safety_margin[safe_mask]
            min_safe_margin = safe_margins.min()
            boundary_mask = safe_mask & torch.isclose(
                safety_margin,
                min_safe_margin,
                rtol=1e-7,
                atol=1e-10,
            )
            B = X_cand[boundary_mask]

        return {
            "S": S,
            "B": B,
            "safe_mask": safe_mask,
            "boundary_mask": boundary_mask,
            "utility_u": utility_u,
            "utility_mean": utility_mean,
            "utility_std": utility_std,
            "mean_f": mean_f,
            "sigma_f": std_f,
            "l_g": l_g,
            "sigma_g": std_g,
            "safety_margin": safety_margin,
            "beta_f": beta_f_width,
            "beta_g": beta_g_width,
        }

    def _expansion_scores(self, sets):
        if self.expansion_uncertainty == "utility":
            return sets["utility_std"][sets["boundary_mask"]]
        sigma_g_boundary = sets["sigma_g"][sets["boundary_mask"]]
        return sigma_g_boundary.amax(dim=(1, 2))

    def suggest(self, num_candidates=4096):
        beta_f_value = torch.as_tensor(self.beta_f_fn(self.n_iter))
        beta_g_value = torch.as_tensor(self.beta_g_fn(self.n_iter))
        beta_f = float(beta_f_value.detach().cpu().item())
        beta_g = float(beta_g_value.detach().cpu().item())

        X_unit = self._sobol_engine.draw(num_candidates).to(
            device=self.device,
            dtype=self.bounds.dtype,
        )
        lb = self.bounds[0]
        ub = self.bounds[1]
        X_cand = lb + (ub - lb) * X_unit

        sets = self._get_sets(X_cand, beta_f=beta_f, beta_g=beta_g)
        retried_locally = False
        retry_source = None

        if sets["S"].shape[0] == 0:
            safe_points = self._observed_safe_points(beta_g)
            retry_source = "certified"
            if safe_points.shape[0] == 0:
                safe_points = self._empirically_safe_observed_points()
                retry_source = "empirical"
                if safe_points.shape[0] == 0:
                    raise RuntimeError(
                        "MultiTaskSafeCtrlBO found no certified or empirically safe point."
                    )
            X_retry = self._local_safe_retry_candidates(safe_points, num_candidates)
            sets = self._get_sets(X_retry, beta_f=beta_f, beta_g=beta_g)
            retried_locally = True

            if sets["S"].shape[0] == 0:
                if retry_source == "empirical":
                    x_next = self._best_empirically_safe_observed_point()
                    return x_next.unsqueeze(0), "empirical_safe_fallback", sets
                x_next = self._best_certified_safe_observed_point(beta_f, beta_g)
                return x_next.unsqueeze(0), "safe_fallback", sets

        if self.bo_steps < self.switch_time:
            scores = self._expansion_scores(sets)
            idx = torch.argmax(scores)
            x_next = sets["B"][idx]
            mode = "expansion"
        else:
            u_S = sets["utility_u"][sets["safe_mask"]]
            idx = torch.argmax(u_S)
            x_next = sets["S"][idx]
            mode = "optimization"

        if retried_locally:
            prefix = "empirical_" if retry_source == "empirical" else ""
            mode = f"{prefix}{mode}_local_retry"

        return x_next.unsqueeze(0), mode, sets

    def observe(
        self,
        x_new,
        y_perf_new,
        y_safe_new,
        train_hypers_every=None,
        training_iter=0,
    ):
        x_new = x_new.to(device=self.device, dtype=self.X.dtype)
        if x_new.dim() == 1:
            x_new = x_new.view(1, -1)
        y_perf_new = self._format_mode_observations(y_perf_new, expected_rows=x_new.shape[0])
        y_safe_new = self._format_safety_observations(y_safe_new, expected_rows=x_new.shape[0])
        if y_safe_new.shape[2] != self.num_safety_constraints:
            raise ValueError(
                f"Expected {self.num_safety_constraints} safety constraints, "
                f"got {y_safe_new.shape[2]}."
            )

        self.X = torch.cat([self.X, x_new], dim=0)
        self.Yf = torch.cat([self.Yf, y_perf_new], dim=0)
        self.Yg = torch.cat([self.Yg, y_safe_new], dim=0)
        self.n_iter += x_new.shape[0]
        self.bo_steps += x_new.shape[0]

        self.model_f.set_train_data(inputs=self.X, targets=self.Yf, strict=False)
        for constraint_idx, model_g in enumerate(self.models_g):
            model_g.set_train_data(
                inputs=self.X,
                targets=self.Yg[:, :, constraint_idx],
                strict=False,
            )

        if (
            train_hypers_every is not None
            and training_iter is not None
            and training_iter > 0
            and int(train_hypers_every) > 0
            and self.bo_steps > 0
            and self.bo_steps % train_hypers_every == 0
        ):
            self._fit_models(training_iter=training_iter)

    def _mode_indices(self, observed_modes):
        mode_to_index = {name: idx for idx, name in enumerate(self.mode_names)}
        indices = []
        for mode in observed_modes:
            if isinstance(mode, str):
                if mode not in mode_to_index:
                    raise ValueError(f"Unknown mode '{mode}'. Expected one of {self.mode_names}.")
                idx = mode_to_index[mode]
            else:
                idx = int(mode)
                if idx < 0 or idx >= self.num_modes:
                    raise ValueError(f"Mode index {idx} is out of range for {self.num_modes} modes.")
            if idx in indices:
                raise ValueError("observed_modes must not contain duplicates.")
            indices.append(idx)
        if not indices:
            raise ValueError("observed_modes must contain at least one mode.")
        return indices

    def _format_partial_mode_observations(self, values, expected_rows, num_observed_modes):
        y = torch.as_tensor(values, dtype=self.X.dtype, device=self.device)
        if y.dim() == 1 and expected_rows == 1 and y.numel() == num_observed_modes:
            y = y.view(1, num_observed_modes)
        elif y.dim() != 2:
            raise ValueError(
                f"Partial mode observations must have shape (n, {num_observed_modes}) "
                f"or ({num_observed_modes},) for one row."
            )
        if y.shape != (expected_rows, num_observed_modes):
            raise ValueError(
                f"Expected partial mode observations with shape "
                f"({expected_rows}, {num_observed_modes}), got {tuple(y.shape)}."
            )
        return y

    def _format_partial_safety_observations(self, values, expected_rows, num_observed_modes):
        y = torch.as_tensor(values, dtype=self.X.dtype, device=self.device)
        if y.dim() == 1 and expected_rows == 1 and self.num_safety_constraints == 1:
            if y.numel() != num_observed_modes:
                raise ValueError(
                    f"Expected {num_observed_modes} partial safety observations, got {y.numel()}."
                )
            y = y.view(1, num_observed_modes, 1)
        elif y.dim() == 2:
            if expected_rows == 1 and y.shape[0] == num_observed_modes:
                y = y.view(1, num_observed_modes, y.shape[1])
            elif y.shape == (expected_rows, num_observed_modes) and self.num_safety_constraints == 1:
                y = y.unsqueeze(-1)
            else:
                raise ValueError(
                    "2D partial safety observations must be (num_observed_modes, k) for one row "
                    "or (n, num_observed_modes) for one safety constraint."
                )
        elif y.dim() != 3:
            raise ValueError("Partial safety observations must be 1D, 2D, or 3D.")

        expected_shape = (expected_rows, num_observed_modes, self.num_safety_constraints)
        if y.shape != expected_shape:
            raise ValueError(f"Expected partial safety observations with shape {expected_shape}, got {tuple(y.shape)}.")
        return y

    def _missing_perf_fill_value(self, override):
        value = self.missing_perf_value if override is None else override
        if value is None:
            value = float(self.Yf.min().detach().cpu().item()) - 1.0
        return float(value)

    def _missing_safety_fill_value(self, override):
        value = self.missing_safety_value if override is None else override
        if value is None:
            value = float(self.safety_thresholds.min().detach().cpu().item()) - 1.0
        return float(value)

    def observe_partial(
        self,
        x_new,
        y_perf_new,
        y_safe_new,
        observed_modes,
        missing_reason="unsafe_abort",
        missing_perf_value=None,
        missing_safety_value=None,
        train_hypers_every=None,
        training_iter=0,
    ):
        """
        Observe an aborted rollout with conservative missing-mode fill.

        This path is intended for unsafe aborts, where unobserved downstream
        modes should not be treated as safe. For no-contact trajectories,
        logging failures, or other non-abort missing data, use a future
        ragged/masked observation path instead of injecting artificial unsafe
        labels.
        """
        if missing_reason not in {"unsafe_abort", "conservative_fill"}:
            raise NotImplementedError(
                "observe_partial currently supports only unsafe-abort conservative fill. "
                "Use complete observations, independent per-mode models, or a future "
                "ragged/masked observation path for no-contact or sensor-missing data."
            )

        x_new = x_new.to(device=self.device, dtype=self.X.dtype)
        if x_new.dim() == 1:
            x_new = x_new.view(1, -1)

        mode_indices = self._mode_indices(observed_modes)
        y_perf_new = self._format_partial_mode_observations(
            y_perf_new,
            expected_rows=x_new.shape[0],
            num_observed_modes=len(mode_indices),
        )
        y_safe_new = self._format_partial_safety_observations(
            y_safe_new,
            expected_rows=x_new.shape[0],
            num_observed_modes=len(mode_indices),
        )

        full_perf = torch.full(
            (x_new.shape[0], self.num_modes),
            self._missing_perf_fill_value(missing_perf_value),
            dtype=self.X.dtype,
            device=self.device,
        )
        full_safe = torch.full(
            (x_new.shape[0], self.num_modes, self.num_safety_constraints),
            self._missing_safety_fill_value(missing_safety_value),
            dtype=self.X.dtype,
            device=self.device,
        )
        full_perf[:, mode_indices] = y_perf_new
        full_safe[:, mode_indices, :] = y_safe_new

        self.observe(
            x_new=x_new,
            y_perf_new=full_perf,
            y_safe_new=full_safe,
            train_hypers_every=train_hypers_every,
            training_iter=training_iter,
        )
