from typing import Any, Optional

import torch
import torch.nn as nn


class Kernel(nn.Module):
    def __init__(
        self,
        metric: str,
        gamma: Optional[float] = None,
        coef0: Optional[float] = None,
        degree: Optional[float] = None,
    ):
        super().__init__()
        if metric not in ["linear", "poly", "polynomial", "rbf", "sigmoid"]:
            raise ValueError(f"unsupported kernel metric: {metric}")
        self.metric = metric
        self.has_gamma = gamma is not None
        self.gamma = float(gamma or 0.0)
        self.has_coef0 = coef0 is not None
        self.coef0 = float(coef0 or 0.0)
        self.has_degree = degree is not None
        self.degree = float(degree or 0.0)

    @classmethod
    def wrap(self, estimator: Any) -> "Kernel":
        if not isinstance(estimator.kernel, str):
            raise ValueError(f"kernel must be str, but got {estimator.kernel}")
        return Kernel(
            metric=estimator.kernel,
            gamma=estimator.gamma
            if not hasattr(estimator, "_gamma")
            else estimator._gamma,
            coef0=estimator.coef0,
            degree=estimator.degree,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.metric == "linear":
            return x @ y.t()
        elif self.metric in ["poly", "polynomial"]:
            return (self._gamma(x) * x @ y.t() + self._coef0()) ** self._degree()
        elif self.metric == "rbf":
            x_norm = ((x ** 2).sum(-1))[:, None]
            y_norm = ((y ** 2).sum(-1))[None]
            dots = x @ y.t()
            dists = (x_norm + y_norm - 2 * dots).clamp_min(0)
            return (-self._gamma(x) * dists).exp()
        elif self.metric == "sigmoid":
            return (self._gamma(x) * x @ y.t() + self._coef0()).tanh()
        raise RuntimeError(f"unsupported kernel: {self.metric}")

    def _gamma(self, x: torch.Tensor) -> float:
        if self.has_gamma:
            return self.gamma
        elif self.metric in ["rbf", "sigmoid", "poly", "polynomial"]:
            return 1.0 / x.shape[1]
        raise RuntimeError(f"unknown default gamma for kernel: {self.metric}")

    def _coef0(self) -> float:
        if self.has_coef0:
            return self.coef0
        elif self.metric in ["sigmoid", "poly", "polynomial"]:
            return 1.0
        raise RuntimeError(f"unknown default coef0 for kernel: {self.metric}")

    def _degree(self) -> float:
        if self.has_degree:
            return self.degree
        elif self.metric in ["poly", "polynomial"]:
            return 3.0
        raise RuntimeError(f"unknown default degree for kernel: {self.metric}")
