import torch
import numpy as np
from kernels import make_safe_bo_kernel
from model import build_gp, fit_gp

torch.set_default_dtype(torch.double)
device = "cuda" if torch.cuda.is_available() else "cpu"

# x0: (n0, d), perf0: (n0,1), safe0:(n0,1)
x0 = [[0,0,0]]
perf0 = [100]
safe0 = [100]

train_X = torch.from_numpy(x0).to(device=device, dtype=torch.double)
train_Y_perf = torch.from_numpy(perf0).to(device=device, dtype=torch.double)
train_Y_safe = torch.from_numpy(safe0).to(device=device, dtype=torch.double)

kernel = make_safe_bo_kernel(device=device)

# Two GPs
model_f, lik_f, mll_f = build_gp(train_X, train_Y_perf, kernel)
model_g, lik_g, mll_g = build_gp(train_X, train_Y_safe, kernel)

fit_gp(model_f, lik_f, mll_f)
fit_gp(model_g, lik_g, mll_g)