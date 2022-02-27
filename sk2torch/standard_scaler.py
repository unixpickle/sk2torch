from typing import Optional

import torch
import torch.jit
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class TorchStandardScaler(nn.Module):
    def __init__(self, mean: Optional[torch.Tensor], scale: Optional[torch.Tensor]):
        super().__init__()
        self.mean = nn.Parameter(mean if mean is not None else torch.zeros(()))
        self.scale = nn.Parameter(scale if scale is not None else torch.ones(()))

    @classmethod
    def supports_wrap(cls, obj: BaseEstimator) -> bool:
        return isinstance(obj, StandardScaler)

    @classmethod
    def wrap(cls, obj: StandardScaler) -> "TorchStandardScaler":
        return cls(
            mean=torch.from_numpy(obj.mean_) if obj.with_mean else None,
            scale=torch.from_numpy(obj.scale_) if obj.with_std else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.scale

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.scale) + self.mean
