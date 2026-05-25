import copy
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood


class SingleOutputGP(ExactGP):
    def __init__(self, train_X, train_Y, likelihood, covar_module):
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        # we use an additive Gaussian kernel as the covariance
        self.covar_module = covar_module

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class ModeTaskGP(ExactGP):
    def __init__(self, train_X, train_Y, likelihood, covar_module, num_modes, task_rank=1):
        super().__init__(train_X, train_Y, likelihood)
        self.num_modes = int(num_modes)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=self.num_modes)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            covar_module,
            num_tasks=self.num_modes,
            rank=int(task_rank),
        )

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class ModeLMCGP(ExactGP):
    def __init__(self, train_X, train_Y, likelihood, covar_modules, num_modes, task_rank=1):
        super().__init__(train_X, train_Y, likelihood)
        self.num_modes = int(num_modes)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=self.num_modes)
        multitask_components = []
        for covar_module in covar_modules:
            # MultitaskKernel.forward calls data_covar_module.forward directly.
            # Wrapping each physics component in a one-term AdditiveKernel keeps
            # active_dims slicing intact for ScaleKernel/RBFKernel components.
            data_kernel = gpytorch.kernels.AdditiveKernel(copy.deepcopy(covar_module))
            multitask_components.append(
                gpytorch.kernels.MultitaskKernel(
                    data_kernel,
                    num_tasks=self.num_modes,
                    rank=int(task_rank),
                )
            )
        self.covar_module = gpytorch.kernels.AdditiveKernel(*multitask_components)

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultitaskMultivariateNormal(mean_x, covar_x)


def build_gp(train_X, train_Y, base_kernel, noise=1e-4):
    """
    train_X: (n, d) torch.double
    train_Y: (n, 1) torch.double
    base_kernel: AdditiveKernel (usually frozen from DARTS search)
    noise: Gaussian likelihood noise variance (not standard deviation)

    Returns:
        model: SingleOutputGP
        likelihood: GaussianLikelihood
        mll: ExactMarginalLogLikelihood
    """
    device = train_X.device
    dtype = train_X.dtype

    # ensure Y is 1D target vector for ExactGP
    if train_Y.dim() == 2 and train_Y.size(-1) == 1:
        train_Y_flat = train_Y.squeeze(-1)
    else:
        train_Y_flat = train_Y

    # each GP has one deepcopied kernel to avoid hyperparameters sharing
    kernel = copy.deepcopy(base_kernel).to(device=device, dtype=dtype)

    noise_value = None if noise is None else float(noise)
    noise_lower_bound = 1e-4
    if noise_value is not None:
        noise_lower_bound = max(min(noise_value * 0.5, 1e-4), 1e-12)

    likelihood = GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lower_bound)
    ).to(device=device, dtype=dtype)
    # if noise is provided, initialize noise; otherwise keep default gpytorch init
    if noise_value is not None:
        likelihood.initialize(noise=noise_value)

    model = SingleOutputGP(train_X, train_Y_flat, likelihood, kernel).to(device=device, dtype=dtype)

    mll = ExactMarginalLogLikelihood(likelihood, model)

    return model, likelihood, mll


def build_mode_task_gp(train_X, train_Y, base_kernel, num_modes, noise=1e-4, task_rank=1):
    """
    Build an exact multi-task GP over mode-level outputs.

    train_X: (n, d)
    train_Y: (n, num_modes)
    base_kernel: kernel over controller parameters x
    num_modes: number of hybrid/contact modes
    noise: Gaussian likelihood noise variance
    task_rank: rank of the learned task covariance matrix
    """
    device = train_X.device
    dtype = train_X.dtype
    train_Y = torch.as_tensor(train_Y, dtype=dtype, device=device)
    if train_Y.dim() != 2 or train_Y.shape[1] != int(num_modes):
        raise ValueError(
            f"train_Y must have shape (n, {int(num_modes)}), got {tuple(train_Y.shape)}."
        )

    kernel = copy.deepcopy(base_kernel).to(device=device, dtype=dtype)

    noise_value = None if noise is None else float(noise)
    noise_lower_bound = 1e-4
    if noise_value is not None:
        noise_lower_bound = max(min(noise_value * 0.5, 1e-4), 1e-12)

    likelihood = MultitaskGaussianLikelihood(
        num_tasks=int(num_modes),
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lower_bound),
    ).to(device=device, dtype=dtype)
    if noise_value is not None:
        likelihood.initialize(noise=noise_value)

    model = ModeTaskGP(
        train_X,
        train_Y,
        likelihood,
        kernel,
        num_modes=num_modes,
        task_rank=task_rank,
    ).to(device=device, dtype=dtype)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    return model, likelihood, mll


def build_mode_lmc_gp(train_X, train_Y, base_kernels, num_modes, noise=1e-4, task_rank=1):
    """
    Build a physics-guided LMC multi-task GP over mode-level outputs.

    base_kernels is a sequence of latent kernels k_q(x, x'). Each component is
    wrapped in its own MultitaskKernel, giving sum_q B_q k_q.
    """
    device = train_X.device
    dtype = train_X.dtype
    train_Y = torch.as_tensor(train_Y, dtype=dtype, device=device)
    if train_Y.dim() != 2 or train_Y.shape[1] != int(num_modes):
        raise ValueError(
            f"train_Y must have shape (n, {int(num_modes)}), got {tuple(train_Y.shape)}."
        )
    if base_kernels is None:
        raise ValueError("base_kernels must be a non-empty kernel or sequence of kernels.")
    if isinstance(base_kernels, gpytorch.kernels.AdditiveKernel):
        kernels = list(base_kernels.kernels)
    elif isinstance(base_kernels, gpytorch.kernels.Kernel):
        kernels = [base_kernels]
    else:
        kernels = list(base_kernels)
    if len(kernels) == 0:
        raise ValueError("base_kernels must be a non-empty kernel or sequence of kernels.")
    kernels = [copy.deepcopy(kernel).to(device=device, dtype=dtype) for kernel in kernels]

    noise_value = None if noise is None else float(noise)
    noise_lower_bound = 1e-4
    if noise_value is not None:
        noise_lower_bound = max(min(noise_value * 0.5, 1e-4), 1e-12)

    likelihood = MultitaskGaussianLikelihood(
        num_tasks=int(num_modes),
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lower_bound),
    ).to(device=device, dtype=dtype)
    if noise_value is not None:
        likelihood.initialize(noise=noise_value)

    model = ModeLMCGP(
        train_X,
        train_Y,
        likelihood,
        kernels,
        num_modes=num_modes,
        task_rank=task_rank,
    ).to(device=device, dtype=dtype)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    return model, likelihood, mll


def iter_multitask_kernels(covar_module):
    if isinstance(covar_module, gpytorch.kernels.MultitaskKernel):
        yield covar_module
        return
    if isinstance(covar_module, gpytorch.kernels.AdditiveKernel):
        for kernel in covar_module.kernels:
            if isinstance(kernel, gpytorch.kernels.MultitaskKernel):
                yield kernel


def fit_gp(
    model,
    likelihood,
    mll,
    training_iter=200,
    lr=0.05,
    train_kernel=False,
    train_mean=False,
    train_noise=True,
):
    """
    Fit GP hyperparameters with Adam.

    By default (for DARTS-frozen kernels in SafeCtrlBO), we recommend:
        train_kernel=False, train_mean=False, train_noise=True
    so that:
        - kernel hyperparameters (lengthscale/outputscale) stay fixed from DARTS
        - only likelihood noise is adapted online if needed.

    Args:
        model: SingleOutputGP
        likelihood: GaussianLikelihood
        mll: ExactMarginalLogLikelihood
        training_iter: number of optimization steps (0 => skip training)
        lr: learning rate for Adam
        train_kernel: whether to update covar_module parameters
        train_mean: whether to update mean_module parameters
        train_noise: whether to update likelihood noise parameters
    """
    if training_iter is None or training_iter <= 0:
        # nothing to do
        return

    # set requires_grad flags according to training options
    for p in model.covar_module.parameters():
        p.requires_grad = bool(train_kernel)

    for p in model.mean_module.parameters():
        p.requires_grad = bool(train_mean)

    for p in likelihood.parameters():
        p.requires_grad = bool(train_noise)

    # collect parameters that are actually trainable
    params = []
    seen = set()
    for param in list(model.parameters()) + list(likelihood.parameters()):
        if not param.requires_grad or id(param) in seen:
            continue
        params.append(param)
        seen.add(id(param))

    # if no parameter is trainable, exit early
    if len(params) == 0:
        return

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(params, lr=lr)

    train_X = model.train_inputs[0]
    train_Y = model.train_targets

    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y)
        loss.backward()
        optimizer.step()


def fit_mode_task_gp(
    model,
    likelihood,
    mll,
    training_iter=200,
    lr=0.05,
    train_data_kernel=False,
    train_task_covar=True,
    train_mean=False,
    train_noise=True,
):
    """
    Fit a ModeTaskGP while allowing task covariance to be learned separately
    from the data kernel. This keeps frozen controller-parameter kernels usable
    while still learning how modes are correlated.
    """
    if training_iter is None or training_iter <= 0:
        return

    multitask_kernels = list(iter_multitask_kernels(model.covar_module))
    if len(multitask_kernels) == 0:
        raise ValueError("fit_mode_task_gp expected a MultitaskKernel or additive LMC multitask kernel.")

    for multitask_kernel in multitask_kernels:
        for p in multitask_kernel.data_covar_module.parameters():
            p.requires_grad = bool(train_data_kernel)

        for p in multitask_kernel.task_covar_module.parameters():
            p.requires_grad = bool(train_task_covar)

    for p in model.mean_module.parameters():
        p.requires_grad = bool(train_mean)

    for p in likelihood.parameters():
        p.requires_grad = bool(train_noise)

    params = []
    seen = set()
    for param in list(model.parameters()) + list(likelihood.parameters()):
        if not param.requires_grad or id(param) in seen:
            continue
        params.append(param)
        seen.add(id(param))

    if len(params) == 0:
        return

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(params, lr=lr)

    train_X = model.train_inputs[0]
    train_Y = model.train_targets

    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y)
        loss.backward()
        optimizer.step()
