"""Sparse variational Gaussian process (GPyTorch) — the SVGP notebook as a model.

Architecture by default = the notebook: inputs scaled to [-1, 1]
(``ScaleToBounds``), MLP mean (3 x 512, ReLU), ``ScaleKernel(RBF)`` with ARD,
learnable inducing points (5 % of the training rows), Gaussian likelihood,
``PredictiveLogLikelihood`` objective, Adam 1e-3 with ``MultiStepLR([50, 100], 0.5)``,
200 epochs, batches of 1024, float64 on CPU.

The model predicts a mean and a standard deviation (in the transformed target
space; the Forecaster maps them back to MW).
"""
from __future__ import annotations

import math
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.models import ApproximateGP
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.base import BaseModel
from ..training.torch_loop import TorchTask, TorchTrainer
from ..utils.seed import set_seed

DTYPES = {"float64": torch.float64, "float32": torch.float32}


class MLPMean(gpytorch.means.Mean):
    def __init__(self, input_size: int, hidden: tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        layers, prev = [], input_size
        for size in hidden:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class GPModel(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, mean: str = "mlp", hidden=(512, 512, 512),
                 kernel: str = "rbf", ard: bool = True, learn_inducing: bool = True,
                 scale_to_bounds: bool = True):
        distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        strategy = VariationalStrategy(self, inducing_points, distribution,
                                       learn_inducing_locations=learn_inducing)
        super().__init__(strategy)
        input_size = inducing_points.size(-1)
        self.scale_to_bounds = ScaleToBounds(-1.0, 1.0) if scale_to_bounds else None
        if mean == "mlp":
            self.mean_module = MLPMean(input_size, tuple(hidden))
        elif mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            raise ValueError("mean must be 'mlp' or 'constant'")
        ard_dims = input_size if ard else None
        if kernel == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif kernel in ("matern52", "matern32", "matern12"):
            nu = {"matern52": 2.5, "matern32": 1.5, "matern12": 0.5}[kernel]
            base = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=ard_dims)
        else:
            raise ValueError("kernel must be 'rbf', 'matern52', 'matern32' or 'matern12'")
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        if self.scale_to_bounds is not None:
            x = self.scale_to_bounds(x)
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _SVGPTask(TorchTask):
    loss_name = "-log-likelihood"

    def __init__(self, module, likelihood, optimizer, scheduler, mll, device, dtype):
        self.module, self.likelihood = module, likelihood
        self.optimizer, self.scheduler, self.mll = optimizer, scheduler, mll
        self.device, self.dtype = device, dtype

    def _batch(self, batch):
        x, y = batch
        return x.to(self.device, self.dtype), y.to(self.device, self.dtype)

    def train_mode(self):
        self.module.train()
        self.likelihood.train()

    def eval_mode(self):
        self.module.eval()
        self.likelihood.eval()

    def train_step(self, batch):
        x, y = self._batch(batch)
        self.optimizer.zero_grad()
        loss = -self.mll(self.module(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, batch):
        x, y = self._batch(batch)
        return (-self.mll(self.module(x), y)).item()

    def end_epoch(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def weights(self):
        return {"module": {k: v.detach().clone() for k, v in self.module.state_dict().items()},
                "likelihood": {k: v.detach().clone() for k, v in self.likelihood.state_dict().items()}}

    def load_weights(self, state):
        self.module.load_state_dict(state["module"])
        self.likelihood.load_state_dict(state["likelihood"])

    def full_state(self):
        return {"weights": self.weights(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None}

    def load_full_state(self, state):
        self.load_weights(state["weights"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])


class SVGPRegressor(BaseModel):
    """Sparse variational GP with an MLP mean (see the module docstring).

    >>> SVGPRegressor(epochs=200, inducing_fraction=0.05, kernel="rbf", seed=42)
    """

    supports_distribution = True

    def __init__(self, inducing_fraction: float = 0.05, n_inducing: int | None = None, mean: str = "mlp",
                 mlp_hidden: tuple[int, ...] = (512, 512, 512), kernel: str = "rbf", ard: bool = True,
                 learn_inducing: bool = True, scale_to_bounds: bool = True, objective: str = "pll",
                 epochs: int = 200, batch_size: int = 1024, learning_rate: float = 0.001,
                 milestones: tuple[int, ...] = (50, 100), gamma: float = 0.5, patience: int | None = None,
                 dtype: str = "float64", device: str = "cpu", seed: int | None = None,
                 checkpoint_dir: str | None = None, resume: bool = False, verbose: int = 1):
        self.inducing_fraction = inducing_fraction
        self.n_inducing = n_inducing
        self.mean = mean
        self.mlp_hidden = mlp_hidden
        self.kernel = kernel
        self.ard = ard
        self.learn_inducing = learn_inducing
        self.scale_to_bounds = scale_to_bounds
        self.objective = objective
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.gamma = gamma
        self.patience = patience
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        self.verbose = verbose

    # ------------------------------------------------------------- helpers
    def _torch_dtype(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}")
        return DTYPES[self.dtype]

    def _tensor(self, values) -> torch.Tensor:
        array = values.to_numpy(dtype=float) if isinstance(values, (pd.DataFrame, pd.Series)) \
            else np.asarray(values, dtype=float)
        return torch.as_tensor(array, dtype=self._torch_dtype())

    def _build(self, inducing: torch.Tensor):
        module = GPModel(inducing, self.mean, self.mlp_hidden, self.kernel, self.ard,
                         self.learn_inducing, self.scale_to_bounds)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        dtype = self._torch_dtype()
        return module.to(self.device, dtype), likelihood.to(self.device, dtype)

    # ------------------------------------------------------------ training
    def fit(self, X, y, X_val=None, y_val=None) -> "SVGPRegressor":
        set_seed(self.seed)  # torch is imported here, so it is seeded too
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        x_t, y_t = self._tensor(X), self._tensor(y).ravel()
        n = x_t.shape[0]
        n_inducing = self.n_inducing or max(1, math.floor(n * self.inducing_fraction + 1e-9))
        inducing = x_t[torch.randperm(n)[:n_inducing]].clone()
        module, likelihood = self._build(inducing)

        optimizer = torch.optim.Adam([{"params": module.parameters()}, {"params": likelihood.parameters()}],
                                     lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(self.milestones),
                                                         gamma=self.gamma) if self.milestones else None
        if self.objective == "pll":
            mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, module, num_data=n)
        elif self.objective == "elbo":
            mll = gpytorch.mlls.VariationalELBO(likelihood, module, num_data=n)
        else:
            raise ValueError("objective must be 'pll' or 'elbo'")

        train_loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None and len(y_val):
            val_loader = DataLoader(TensorDataset(self._tensor(X_val), self._tensor(y_val).ravel()),
                                    batch_size=self.batch_size, shuffle=False)

        task = _SVGPTask(module, likelihood, optimizer, scheduler, mll, self.device, self._torch_dtype())
        trainer = TorchTrainer(self.epochs, patience=self.patience, checkpoint_dir=self.checkpoint_dir,
                               resume=self.resume, verbose=self.verbose)
        result = trainer.fit(task, train_loader, val_loader)

        self.module_, self.likelihood_ = module, likelihood
        self.weights_ = {"best": result.best_weights, "last": result.last_weights}
        self.input_size_, self.n_inducing_ = int(x_t.shape[1]), int(n_inducing)
        self.history_ = result.history
        self._active_weights = None
        self.fitted_ = True
        self._use_weights("best")
        return self

    def _use_weights(self, weights: str) -> None:
        self._check_weights(weights)
        if getattr(self, "_active_weights", None) != weights:
            state = self.weights_[weights]
            self.module_.load_state_dict(state["module"])
            self.likelihood_.load_state_dict(state["likelihood"])
            self._active_weights = weights

    # ---------------------------------------------------------- prediction
    def predict_distribution(self, X, weights: str = "best") -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and std (noise included), batch by batch."""
        self._check_fitted()
        self._use_weights(weights)
        self.module_.eval()
        self.likelihood_.eval()
        x_t = self._tensor(X)
        means, stds = [], []
        with torch.no_grad():
            for start in range(0, x_t.shape[0], self.batch_size):
                batch = x_t[start:start + self.batch_size].to(self.device)
                dist = self.likelihood_(self.module_(batch))
                means.append(dist.mean.cpu().numpy())
                stds.append(dist.stddev.cpu().numpy())
        if not means:
            return np.empty(0), np.empty(0)
        return np.concatenate(means), np.concatenate(stds)

    def predict(self, X, weights: str = "best") -> np.ndarray:
        return self.predict_distribution(X, weights)[0]

    def feature_importance(self) -> pd.Series:
        """1 / ARD lengthscale per input (larger = the GP reacts more to that feature)."""
        self._check_fitted()
        base = self.module_.covar_module.base_kernel
        lengthscale = base.lengthscale.detach().cpu().numpy().ravel()
        names = self.feature_names_ or [f"x{i}" for i in range(lengthscale.size)]
        if lengthscale.size != len(names):
            raise NotImplementedError("Feature importance needs ard=True")
        return pd.Series(1.0 / lengthscale, index=names).sort_values(ascending=False)

    # --------------------------------------------------------------- saving
    def _save_state(self, path: Path) -> None:
        torch.save({"weights": self.weights_, "input_size": self.input_size_, "n_inducing": self.n_inducing_,
                    "feature_names": self.feature_names_}, path / "svgp.pt")

    def _load_state(self, path: Path) -> None:
        state = torch.load(path / "svgp.pt", weights_only=False)
        self.weights_ = state["weights"]
        self.input_size_, self.n_inducing_ = state["input_size"], state["n_inducing"]
        self.feature_names_ = state["feature_names"]
        dummy = torch.zeros(self.n_inducing_, self.input_size_, dtype=self._torch_dtype())
        self.module_, self.likelihood_ = self._build(dummy)
        self._active_weights = None
        self._use_weights("best")
