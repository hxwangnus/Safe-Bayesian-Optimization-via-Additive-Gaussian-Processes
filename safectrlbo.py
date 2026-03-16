# GPyTorch version of SafeCtrlBO. 
# Author: H. Wang, December 2025.
import torch
import gpytorch
# from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine
from model import build_gp, fit_gp


class SafeCtrlBO:
    def __init__(
        self,
        init_X,             # (n0, d)
        init_Y_perf,        # (n0, 1)
        init_Y_safe,        # (n0, 1) or None
        bounds,             # (2, d) tensor [[l1..ld],[u1..ud]]
        base_kernel,        # AdditiveKernel (frozen from DARTS search)
        safety_threshold=None,   # h_s
        switch_time=15,     # T0
        beta_fn=None,
        tau=0.1,
        device="cpu",
        init_training_iter=0,  # number of training steps at initialization (0 => use DARTS hyper as-is)
        sobol_seed=None,
    ):
        self.device = device
        self.bounds = bounds.to(device)

        # whether we have a separate safety signal g(x)
        self.use_safety = (init_Y_safe is not None) and (safety_threshold is not None)
        self.safety_threshold = safety_threshold

        self.switch_time = switch_time
        self.beta_fn = beta_fn or (lambda n: 2.0 * torch.log(torch.tensor(float(n + 1.0))))
        self.tau = tau
        self._sobol_engine = SobolEngine(
            dimension=self.bounds.shape[1],
            scramble=True,
            seed=sobol_seed,
        )

        self.X = init_X.to(device)
        self.Yf = init_Y_perf.to(device)
        self.Yg = init_Y_safe.to(device) if self.use_safety else None

        # current number of observations (used for beta_t etc.)
        self.n_iter = self.X.shape[0]

        # build two GPs with a frozen additive kernel learned by DARTS
        self.rebuild_models(base_kernel, training_iter=init_training_iter)

    def rebuild_models(self, base_kernel, training_iter=0):
        """
        Build GP models for f and g using the same frozen base_kernel.
        If training_iter > 0, fit_gp can be used to slightly refine noise or
        (optionally) kernel hyperparameters; with DARTS, we typically set
        training_iter=0 to keep the learned kernel unchanged.
        """
        self.model_f, self.lik_f, self.mll_f = build_gp(
            self.X, self.Yf, base_kernel
        )

        if self.use_safety:
            self.model_g, self.lik_g, self.mll_g = build_gp(
                self.X, self.Yg, base_kernel
            )
        else:
            self.model_g = None
            self.lik_g = None
            self.mll_g = None

        if training_iter is not None and training_iter > 0:
            fit_gp(self.model_f, self.lik_f, self.mll_f, training_iter=training_iter)
            if self.use_safety:
                fit_gp(self.model_g, self.lik_g, self.mll_g, training_iter=training_iter)

    @torch.no_grad()
    def posterior_mean_std(self, model, likelihood, Xtest):
        model.eval()
        likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            pred = likelihood(model(Xtest))
        mean = pred.mean
        std = pred.variance.sqrt()
        return mean, std

    def _get_sets(self, X_cand, beta):
        """
        Calculate Sn, Bn, u_f (UCB of f), sigma_f, l_g (LCB of g)

        If self.use_safety is False, we fall back to unconstrained BO:
        S = B = all candidates, and l_g is a dummy tensor.
        """
        X_cand = X_cand.to(self.device)     # set of parameter candidates

        # posterior of f
        mu_f, std_f = self.posterior_mean_std(self.model_f, self.lik_f, X_cand)

        beta_sqrt = torch.sqrt(torch.tensor(beta, dtype=X_cand.dtype, device=self.device))
        u_f = mu_f + beta_sqrt * std_f

        if not self.use_safety:
            # unconstrained case: everything is "safe"
            safe_mask = torch.ones(X_cand.size(0), dtype=torch.bool, device=self.device)
            boundary_mask = safe_mask.clone()
            S = X_cand
            B = X_cand
            # dummy l_g just for API compatibility
            l_g = torch.full_like(mu_f, fill_value=0.0)
            return {
                "S": S,
                "B": B,
                "safe_mask": safe_mask,
                "boundary_mask": boundary_mask,
                "u_f": u_f,
                "sigma_f": std_f,
                "l_g": l_g,
            }

        # posterior of g (safety) in constrained case
        mu_g, std_g = self.posterior_mean_std(self.model_g, self.lik_g, X_cand)
        l_g = mu_g - beta_sqrt * std_g

        # safe set Sn
        safe_mask = l_g >= self.safety_threshold
        S = X_cand[safe_mask]
        if S.numel() == 0:
            # if safe set is empty, use the whole set of parameters
            # this is possible if X_cand is sparse and small
            S = X_cand
            safe_mask = torch.ones_like(l_g, dtype=torch.bool)

        # safe boundary set Bn
        boundary_mask = safe_mask & (torch.abs(l_g - self.safety_threshold) <= self.tau)
        B = X_cand[boundary_mask]
        if B.numel() == 0:
            # if boundary set is empty, use the safe set
            B = S
            boundary_mask = safe_mask

        return {
            "S": S,
            "B": B,
            "safe_mask": safe_mask,
            "boundary_mask": boundary_mask,
            "u_f": u_f,
            "sigma_f": std_f,
            "l_g": l_g,
        }

    def suggest(self, num_candidates=4096):
        """
        generate next parameter x_next within [bounds]
        previously (in GPy) called as:
        x_next = opt.optimize()
        """
        # here n_iter is the current number of observations;
        # you can also use (self.n_iter + 1) if you prefer beta_{t+1}
        beta = float(self.beta_fn(self.n_iter))

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

        if self.n_iter <= self.switch_time:
            # Safe exploration, maximize sigma_f in Bn
            sigma_B = sets["sigma_f"][sets["boundary_mask"]]
            idx = torch.argmax(sigma_B)
            x_next = sets["B"][idx]
            mode = "expansion"
        else:
            # Exploitation, maximize UCB_f in S_n
            u_S = sets["u_f"][sets["safe_mask"]]
            idx = torch.argmax(u_S)
            x_next = sets["S"][idx]
            mode = "optimization"

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
        x_new = x_new.to(self.device)
        y_perf_new = torch.as_tensor(
            y_perf_new, dtype=self.X.dtype, device=self.device
        ).view(-1, 1)

        self.X = torch.cat([self.X, x_new], dim=0)
        self.Yf = torch.cat([self.Yf, y_perf_new], dim=0)

        if self.use_safety:
            y_safe_new = torch.as_tensor(
                y_safe_new, dtype=self.X.dtype, device=self.device
            ).view(-1, 1)
            self.Yg = torch.cat([self.Yg, y_safe_new], dim=0)

        # increase number of observations
        self.n_iter += 1

        # update train data (no change to kernel structure / hyperparameters here)
        self.model_f.set_train_data(
            inputs=self.X, targets=self.Yf.squeeze(-1), strict=False
        )
        if self.use_safety:
            self.model_g.set_train_data(
                inputs=self.X, targets=self.Yg.squeeze(-1), strict=False
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
                fit_gp(self.model_g, self.lik_g, self.mll_g,
                       training_iter=training_iter,
                       train_kernel=False,
                       train_mean=False,
                       train_noise=True)
