import torch
from gpytorch.kernels import AdditiveKernel, ProductKernel, RBFKernel, ScaleKernel


def _make_scaled_product_rbf(dims, lengthscales, outputscale):
    if len(dims) != len(lengthscales):
        raise ValueError("dims and lengthscales must have the same length.")

    factors = []
    for dim, lengthscale in zip(dims, lengthscales):
        factor = RBFKernel(active_dims=(int(dim),))
        factor.initialize(lengthscale=float(lengthscale))
        factors.append(factor)

    kernel = factors[0]
    for factor in factors[1:]:
        kernel = ProductKernel(kernel, factor)

    scaled_kernel = ScaleKernel(kernel)
    scaled_kernel.initialize(outputscale=float(outputscale))
    return scaled_kernel


def make_safe_bo_kernel(dtype=torch.double, device="cpu"):
    """
    Build the frozen additive kernel exported from the DARTS search.

    Each component is an RBF product over a subset of input dimensions with its
    effective outputscale already folded into the component weight.
    """
    components = [
        _make_scaled_product_rbf(
            dims=(0, 1, 2),
            lengthscales=(0.382012, 4.97979, 1.35749),
            outputscale=0.110905,
        ),
        _make_scaled_product_rbf(
            dims=(1, 2),
            lengthscales=(0.173138, 5.18369),
            outputscale=0.0251577,
        ),
        _make_scaled_product_rbf(
            dims=(0,),
            lengthscales=(0.326926,),
            outputscale=0.0168713,
        ),
        _make_scaled_product_rbf(
            dims=(0, 1),
            lengthscales=(2.9574, 2.1265),
            outputscale=0.00637354,
        ),
        _make_scaled_product_rbf(
            dims=(2,),
            lengthscales=(2.7951,),
            outputscale=0.00517552,
        ),
    ]
    return AdditiveKernel(*components).to(device=device, dtype=dtype)
