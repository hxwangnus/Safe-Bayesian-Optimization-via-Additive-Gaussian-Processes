import torch

from kernels import make_safe_bo_kernel
from model import build_gp, fit_gp


torch.set_default_dtype(torch.double)


def build_initial_models(device=None):
    """
    Build a minimal pair of performance/safety GPs for quick sanity checks.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_X = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.double, device=device)
    train_Y_perf = torch.tensor([[100.0]], dtype=torch.double, device=device)
    train_Y_safe = torch.tensor([[100.0]], dtype=torch.double, device=device)

    kernel = make_safe_bo_kernel(device=device, dtype=torch.double)

    model_f, lik_f, mll_f = build_gp(train_X, train_Y_perf, kernel)
    model_g, lik_g, mll_g = build_gp(train_X, train_Y_safe, kernel)

    return (model_f, lik_f, mll_f), (model_g, lik_g, mll_g)


def main():
    (model_f, lik_f, mll_f), (model_g, lik_g, mll_g) = build_initial_models()
    fit_gp(model_f, lik_f, mll_f)
    fit_gp(model_g, lik_g, mll_g)
    print("Initialized performance and safety GPs.")


if __name__ == "__main__":
    main()
