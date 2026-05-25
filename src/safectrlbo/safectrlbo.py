# GPyTorch version of SafeCtrlBO.
# Author: H. Wang, December 2025.
import math

import torch
import gpytorch
# from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine
from .device_utils import resolve_device
from .model import build_gp, fit_gp


class SafeCtrlBO:
    def __init__(
        self,
        init_X,             # (n0, d)
        init_Y_perf,        # (n0, 1)
        init_Y_safe,        # (n0, m), (n0, 1), or None
        bounds,             # (2, d) tensor [[l1..ld],[u1..ud]]
        base_kernel,        # AdditiveKernel (frozen from DARTS search)
        safety_threshold=None,   # scalar h_s or length-m thresholds
        switch_time=15,     # T0
        beta_fn=None,
        tau=0.1,
        device="cpu",
        init_training_iter=0,  # number of training steps at initialization (0 => use DARTS hyper as-is)
        likelihood_noise=1e-4,  # Gaussian likelihood noise variance used by both GPs
        sobol_seed=None,
        safe_retry_radius=0.05,
        rkhs_bound=1.0,
        noise_bound=None,
        delta=0.05,
        information_gain_fn=None,
        expansion_uncertainty="safety",
    ):
        self.device = resolve_device(device)
        self.bounds = bounds.to(self.device)

        if (init_Y_safe is None) != (safety_threshold is None):
            raise ValueError(
                "init_Y_safe and safety_threshold must either both be provided "
                "or both be None for unconstrained BO."
            )

        if expansion_uncertainty not in {"safety", "objective"}:
            raise ValueError("expansion_uncertainty must be 'safety' or 'objective'.")

        # whether we have one or more separate safety signals g_i(x)
        self.use_safety = init_Y_safe is not None

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
        self.beta_fn = beta_fn or self._default_beta_fn
        self.expansion_uncertainty = expansion_uncertainty
        self._sobol_engine = SobolEngine(
            dimension=self.bounds.shape[1],
            scramble=True,
            seed=sobol_seed,
        )

        self.X = init_X.to(self.device)
        self.Yf = init_Y_perf.to(self.device)
        if self.use_safety:
            self.Yg = self._format_safety_observations(init_Y_safe, expected_rows=self.X.shape[0])
            self.num_safety_constraints = self.Yg.shape[1]
            self.safety_thresholds = self._format_safety_thresholds(
                safety_threshold,
                self.num_safety_constraints,
            )
            self.safety_threshold = (
                self.safety_thresholds[0]
                if self.num_safety_constraints == 1
                else self.safety_thresholds
            )
        else:
            self.Yg = None
            self.num_safety_constraints = 0
            self.safety_thresholds = None
            self.safety_threshold = None

        # current number of observations (used for beta_t etc.)
        self.n_iter = self.X.shape[0]

        # build two GPs with a frozen additive kernel learned by DARTS
        self.rebuild_models(base_kernel, training_iter=init_training_iter)

    def _default_information_gain(self, t):
        t_value = max(int(t), 0)
        return torch.log(torch.tensor(float(t_value + 1.0), device=self.device))

    def _default_beta_fn(self, n):
        """
        Paper-style confidence width:
            beta_t = B + R * sqrt(2 * (gamma_{t-1} + 1 + log(1 / delta))).

        The RKHS bound B, sub-Gaussian noise bound R, and information gain
        approximation are user-configurable because practical safety depends on
        their calibration.
        """
        t_minus_one = max(int(n) - 1, 0)
        gamma = torch.as_tensor(
            self.information_gain_fn(t_minus_one),
            dtype=self.X.dtype,
            device=self.device,
        )
        confidence = 2.0 * (gamma + 1.0 + math.log(1.0 / self.delta))
        return self.rkhs_bound + self.noise_bound * torch.sqrt(confidence)

    def _format_safety_observations(self, values, expected_rows):
        y_safe = torch.as_tensor(values, dtype=self.X.dtype, device=self.device)
        if y_safe.dim() == 0:
            y_safe = y_safe.view(1, 1)
        elif y_safe.dim() == 1:
            if y_safe.numel() == expected_rows:
                y_safe = y_safe.view(expected_rows, 1)
            elif expected_rows == 1:
                y_safe = y_safe.view(1, -1)
            else:
                raise ValueError(
                    "1D safety observations are ambiguous: expected one value per "
                    "row or a single-row vector of constraints."
                )
        elif y_safe.dim() == 2:
            if y_safe.shape[0] != expected_rows:
                raise ValueError(
                    f"Expected {expected_rows} rows of safety observations, got {y_safe.shape[0]}."
                )
        else:
            raise ValueError("Safety observations must be scalar, 1D, or 2D.")

        if y_safe.shape[0] != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows of safety observations, got {y_safe.shape[0]}."
            )
        return y_safe

    def _format_safety_thresholds(self, safety_threshold, num_constraints):
        thresholds = torch.as_tensor(
            safety_threshold,
            dtype=self.X.dtype,
            device=self.device,
        )
        if thresholds.dim() == 0 or thresholds.numel() == 1:
            thresholds = thresholds.reshape(1).expand(num_constraints)
        else:
            thresholds = thresholds.reshape(-1)
            if thresholds.numel() != num_constraints:
                raise ValueError(
                    f"Expected {num_constraints} safety thresholds, got {thresholds.numel()}."
                )
        return thresholds

    def rebuild_models(self, base_kernel, training_iter=0):
        """
        Build GP models for f and all g_i using the same frozen base_kernel.
        If training_iter > 0, fit_gp can be used to slightly refine noise or
        (optionally) kernel hyperparameters; with DARTS, we typically set
        training_iter=0 to keep the learned kernel unchanged.
        """
        self.model_f, self.lik_f, self.mll_f = build_gp(
            self.X, self.Yf, base_kernel, noise=self.likelihood_noise
        )

        if self.use_safety:
            self.models_g = []
            self.liks_g = []
            self.mlls_g = []
            for constraint_idx in range(self.num_safety_constraints):
                model_g, lik_g, mll_g = build_gp(
                    self.X,
                    self.Yg[:, constraint_idx:constraint_idx + 1],
                    base_kernel,
                    noise=self.likelihood_noise,
                )
                self.models_g.append(model_g)
                self.liks_g.append(lik_g)
                self.mlls_g.append(mll_g)

            # Backward-compatible aliases for existing single-constraint callers.
            if self.num_safety_constraints == 1:
                self.model_g = self.models_g[0]
                self.lik_g = self.liks_g[0]
                self.mll_g = self.mlls_g[0]
            else:
                self.model_g = self.models_g
                self.lik_g = self.liks_g
                self.mll_g = self.mlls_g
        else:
            self.models_g = []
            self.liks_g = []
            self.mlls_g = []
            self.model_g = None
            self.lik_g = None
            self.mll_g = None

        if training_iter is not None and training_iter > 0:
            fit_gp(self.model_f, self.lik_f, self.mll_f, training_iter=training_iter)
            if self.use_safety:
                for model_g, lik_g, mll_g in zip(self.models_g, self.liks_g, self.mlls_g):
                    fit_gp(model_g, lik_g, mll_g, training_iter=training_iter)

    @torch.no_grad()
    def posterior_mean_std(self, model, likelihood, Xtest):
        """
        Return the posterior over the latent GP function values.

        BO confidence bounds should be built from epistemic uncertainty in the
        latent function, not from the observation-noise distribution.
        """
        model.eval()
        if likelihood is not None:
            likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            pred = model(Xtest)
        mean = pred.mean
        std = pred.variance.clamp_min(0.0).sqrt()
        return mean, std

    def _beta_width(self, beta, dtype):
        width = torch.as_tensor(beta, dtype=dtype, device=self.device)
        if torch.any(width < 0):
            raise ValueError("beta confidence width must be non-negative.")
        return width

    def _safety_posterior_mean_std(self, Xtest):
        if not self.use_safety:
            empty = torch.empty((Xtest.shape[0], 0), dtype=Xtest.dtype, device=self.device)
            return empty, empty

        means = []
        stds = []
        for model_g, lik_g in zip(self.models_g, self.liks_g):
            mean_g, std_g = self.posterior_mean_std(model_g, lik_g, Xtest)
            means.append(mean_g.reshape(-1))
            stds.append(std_g.reshape(-1))
        return torch.stack(means, dim=-1), torch.stack(stds, dim=-1)

    def _observed_safe_points(self, beta):
        """
        Return observed points that are still certified safe under all safety GPs.
        """
        if not self.use_safety:
            return self.X

        beta_width = self._beta_width(beta, self.X.dtype)
        mu_g_obs, std_g_obs = self._safety_posterior_mean_std(self.X)
        l_g_obs = mu_g_obs - beta_width * std_g_obs
        safe_obs_mask = torch.all(l_g_obs >= self.safety_thresholds.view(1, -1), dim=-1)
        return self.X[safe_obs_mask]

    def _empirically_safe_observed_points(self):
        """
        Return observed points whose measured safety values satisfy the threshold.
        """
        if not self.use_safety:
            return self.X

        safe_obs_mask = torch.all(self.Yg >= self.safety_thresholds.view(1, -1), dim=-1)
        return self.X[safe_obs_mask]

    def _local_safe_retry_candidates(self, safe_points, num_candidates):
        """
        Sample a local cloud around anchor points and include the anchors
        themselves so the retry set always contains the original points.
        """
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
        perturb = torch.randn_like(centers) * perturb_scale
        X_local = centers + perturb

        lb = self.bounds[0].unsqueeze(0)
        ub = self.bounds[1].unsqueeze(0)
        X_local = torch.maximum(torch.minimum(X_local, ub), lb)

        return torch.cat([safe_points, X_local], dim=0)

    def _best_safe_observed_point(self, beta):
        """
        Choose the best certified-safe observed point by the current UCB of f.
        """
        safe_points = self._observed_safe_points(beta)
        if safe_points.numel() == 0:
            raise RuntimeError(
                "SafeCtrlBO could not certify any observed point as safe. "
                "Please provide a safer initialization, relax the safety threshold, "
                "or adjust the GP uncertainty settings."
            )

        beta_width = self._beta_width(beta, safe_points.dtype)
        mu_f_obs, std_f_obs = self.posterior_mean_std(self.model_f, self.lik_f, safe_points)
        u_f_obs = mu_f_obs + beta_width * std_f_obs
        return safe_points[torch.argmax(u_f_obs)]

    def _best_empirically_safe_observed_point(self):
        """
        Choose the best observed point among measurements that satisfied safety.
        """
        safe_obs_mask = torch.all(self.Yg >= self.safety_thresholds.view(1, -1), dim=-1)
        if not torch.any(safe_obs_mask):
            raise RuntimeError(
                "SafeCtrlBO has no observed measurement that satisfies the safety threshold."
            )

        safe_points = self.X[safe_obs_mask]
        safe_perf = self.Yf.squeeze(-1)[safe_obs_mask]
        return safe_points[torch.argmax(safe_perf)]

    def _get_sets(self, X_cand, beta):
        """
        Calculate Sn, Bn, u_f (UCB of f), sigma_f, l_g (LCB of g_i)

        If self.use_safety is False, we fall back to unconstrained BO:
        S = B = all candidates, and l_g is a dummy tensor.
        """
        X_cand = X_cand.to(self.device)     # set of parameter candidates

        # posterior of f
        mu_f, std_f = self.posterior_mean_std(self.model_f, self.lik_f, X_cand)

        beta_width = self._beta_width(beta, X_cand.dtype)
        u_f = mu_f + beta_width * std_f

        if not self.use_safety:
            # unconstrained case: everything is "safe"
            safe_mask = torch.ones(X_cand.size(0), dtype=torch.bool, device=self.device)
            boundary_mask = safe_mask.clone()
            S = X_cand
            B = X_cand
            # dummy safety tensors just for API compatibility
            l_g = torch.empty((X_cand.size(0), 0), dtype=X_cand.dtype, device=self.device)
            sigma_g = torch.empty((X_cand.size(0), 0), dtype=X_cand.dtype, device=self.device)
            safety_margin = torch.full((X_cand.size(0),), float("inf"), dtype=X_cand.dtype, device=self.device)
            return {
                "S": S,
                "B": B,
                "safe_mask": safe_mask,
                "boundary_mask": boundary_mask,
                "u_f": u_f,
                "sigma_f": std_f,
                "l_g": l_g,
                "sigma_g": sigma_g,
                "safety_margin": safety_margin,
            }

        # posterior of all g_i (safety) in constrained case
        mu_g, std_g = self._safety_posterior_mean_std(X_cand)
        l_g = mu_g - beta_width * std_g

        # safe set Sn
        thresholds = self.safety_thresholds.to(dtype=X_cand.dtype).view(1, -1)
        per_constraint_margin = l_g - thresholds
        safety_margin = per_constraint_margin.min(dim=-1).values
        safe_mask = torch.all(per_constraint_margin >= 0.0, dim=-1)
        S = X_cand[safe_mask]

        # safe boundary set Bn
        tau = torch.as_tensor(self.tau, dtype=X_cand.dtype, device=self.device)
        boundary_mask = safe_mask & (safety_margin <= tau)
        B = X_cand[boundary_mask]
        if B.shape[0] == 0 and S.shape[0] > 0:
            # If the relaxed boundary is empty, use the safe candidates with
            # the smallest lower-confidence safety margin, as in Eq. 15.
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
            "u_f": u_f,
            "sigma_f": std_f,
            "l_g": l_g,
            "sigma_g": std_g,
            "safety_margin": safety_margin,
        }

    def _expansion_scores(self, sets):
        if (
            not self.use_safety
            or self.expansion_uncertainty == "objective"
            or sets["sigma_g"].numel() == 0
        ):
            return sets["sigma_f"][sets["boundary_mask"]]

        sigma_g_boundary = sets["sigma_g"][sets["boundary_mask"]]
        return sigma_g_boundary.max(dim=-1).values

    def suggest(self, num_candidates=4096):
        """
        generate next parameter x_next within [bounds]
        previously (in GPy) called as:
        x_next = opt.optimize()
        """
        # here n_iter is the current number of observations;
        # you can also use (self.n_iter + 1) if you prefer beta_{t+1}
        beta_value = torch.as_tensor(self.beta_fn(self.n_iter))
        beta = float(beta_value.detach().cpu().item())

        # # sample the candidates in the box (Sobol)
        # # sample n set(s) of points
        # # each set with "number of candidates" points with dimension d
        # # 4096 candidate points after squeeze(0)
        # # time complexity is O(n*q), n is the num of observed data, q is num of candidates
        # X_cand = draw_sobol_samples(
        #     bounds=self.bounds,
        #     n=1,
        #     q=num_candidates,
        # ).squeeze(0).to(self.device)

        # Use SobolEngine instead, to avoid BoTorch
        # SobolEngine draws points in [0,1]^d, then we affine-transform them to [l_i, u_i]
        # time complexity is O(n*q), n is the num of observed data, q is num of candidates
        # shape: (num_candidates, d), values in [0, 1]
        X_unit = self._sobol_engine.draw(num_candidates).to(
            device=self.device,
            dtype=self.bounds.dtype,
        )

        lb = self.bounds[0]  # (d,)
        ub = self.bounds[1]  # (d,)
        X_cand = lb + (ub - lb) * X_unit  # (num_candidates, d)

        sets = self._get_sets(X_cand, beta)
        retried_locally = False
        retry_source = None

        if self.use_safety and sets["S"].shape[0] == 0:
            safe_points = self._observed_safe_points(beta)
            retry_source = "certified"
            if safe_points.shape[0] == 0:
                safe_points = self._empirically_safe_observed_points()
                retry_source = "empirical"
                if safe_points.shape[0] == 0:
                    raise RuntimeError(
                        "SafeCtrlBO found no certified-safe candidate and no observed "
                        "measurement that satisfied the safety threshold."
                    )

            X_retry = self._local_safe_retry_candidates(safe_points, num_candidates)
            sets = self._get_sets(X_retry, beta)
            retried_locally = True

            if sets["S"].shape[0] == 0:
                if retry_source == "empirical":
                    x_next = self._best_empirically_safe_observed_point()
                    return x_next.unsqueeze(0), "empirical_safe_fallback", sets

                x_next = self._best_safe_observed_point(beta)
                return x_next.unsqueeze(0), "safe_fallback", sets

        if self.n_iter <= self.switch_time:
            # Safe exploration over Bn. By default this follows Algorithm 1:
            # maximize the largest safety uncertainty max_i sigma_g_i. The
            # objective-uncertainty variant is kept for ablation studies.
            expansion_scores = self._expansion_scores(sets)
            idx = torch.argmax(expansion_scores)
            x_next = sets["B"][idx]
            mode = "expansion"
        else:
            # Exploitation, maximize UCB_f in S_n
            u_S = sets["u_f"][sets["safe_mask"]]
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
        y_safe_new=None,
        train_hypers_every=None,
        training_iter=0,
    ):
        """
        Add new observation and (optionally) re-train GP.
        x_new in the shape (1,d)
        y_*_new in the shape (1,1) or a scalar

        With a DARTS-learned frozen kernel, we typically:
          - always update train data via set_train_data
          - optionally update only the likelihood noise in fit_gp
            every 'train_hypers_every' iterations (e.g., to adapt noise).
        """
        # new observation
        x_new = x_new.to(device=self.device, dtype=self.X.dtype)
        if x_new.dim() == 1:
            x_new = x_new.view(1, -1)
        y_perf_new = torch.as_tensor(
            y_perf_new, dtype=self.X.dtype, device=self.device
        ).view(-1, 1)

        self.X = torch.cat([self.X, x_new], dim=0)
        self.Yf = torch.cat([self.Yf, y_perf_new], dim=0)

        if self.use_safety:
            if y_safe_new is None:
                raise ValueError("y_safe_new must be provided when safety constraints are enabled.")
            y_safe_new = self._format_safety_observations(
                y_safe_new,
                expected_rows=x_new.shape[0],
            )
            if y_safe_new.shape[1] != self.num_safety_constraints:
                raise ValueError(
                    f"Expected {self.num_safety_constraints} safety values per row, "
                    f"got {y_safe_new.shape[1]}."
                )
            self.Yg = torch.cat([self.Yg, y_safe_new], dim=0)

        # increase number of observations
        self.n_iter += x_new.shape[0]

        # update train data (no change to kernel structure / hyperparameters here)
        self.model_f.set_train_data(
            inputs=self.X, targets=self.Yf.squeeze(-1), strict=False
        )
        if self.use_safety:
            for constraint_idx, model_g in enumerate(self.models_g):
                model_g.set_train_data(
                    inputs=self.X,
                    targets=self.Yg[:, constraint_idx],
                    strict=False,
                )

        # optimize hyper-parameters (e.g., noise) after K iterations
        if (
            train_hypers_every is not None
            and training_iter is not None
            and training_iter > 0
            and self.n_iter % train_hypers_every == 0
        ):
            fit_gp(self.model_f, self.lik_f, self.mll_f,
                   training_iter=training_iter,
                   train_kernel=False,
                   train_mean=False,
                   train_noise=True)
            if self.use_safety:
                for model_g, lik_g, mll_g in zip(self.models_g, self.liks_g, self.mlls_g):
                    fit_gp(model_g, lik_g, mll_g,
                           training_iter=training_iter,
                           train_kernel=False,
                           train_mean=False,
                           train_noise=True)
