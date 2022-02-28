from typing import Optional

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
        self.metric = metric
        self.gamma = gamma or 0.0
        self.coef0 = coef0 or 0.0
        self.degree = degree or 0.0

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.metric == "linear":
            return x @ y.t()
        elif self.metric == "poly":
            return (self.gamma * x @ y.t() + self.coef0) ** self.degree
        elif self.metric == "rbf":
            x_norm = ((x ** 2).sum(-1))[:, None]
            y_norm = ((y ** 2).sum(-1))[None]
            dots = x @ y.t()
            dists = (x_norm + y_norm - 2 * dots).clamp_min(0)
            return (-self.gamma * dists).exp()
        elif self.metric == "sigmoid":
            return (self.gamma * x @ y.t() + self.coef0).tanh()
        raise RuntimeError(f"unsupported kernel: {self.metric}")
