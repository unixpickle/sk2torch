from typing import List, Optional, Type

import torch
import torch.jit
import torch.nn as nn
from sklearn.decomposition import PCA


class TorchPCA(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        scale: Optional[torch.Tensor],
    ):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.components = nn.Parameter(components)
        if scale is not None:
            self.scale = nn.Parameter(scale)
        else:
            self.scale = 1.0

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [PCA]

    @classmethod
    def wrap(cls, obj: PCA) -> "TorchPCA":
        components = torch.from_numpy(obj.components_)
        mean = obj.mean_
        if mean is None:
            mean = torch.zeros(
                components.shape[1], device=components.device, dtype=components.dtype
            )
        else:
            mean = torch.from_numpy(mean)
        explained_variance = torch.from_numpy(obj.explained_variance_)
        return cls(
            mean=mean,
            components=components,
            scale=explained_variance.rsqrt() if obj.whiten else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) @ self.components.t()) * self.scale

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return ((x / self.scale) @ self.components) + self.mean
