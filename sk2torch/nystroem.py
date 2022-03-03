from typing import List, Type

import torch
import torch.jit
import torch.nn as nn
from sklearn.kernel_approximation import Nystroem

from .kernel import Kernel


class TorchNystroem(nn.Module):
    def __init__(
        self, kernel: Kernel, components: torch.Tensor, normalization: torch.Tensor
    ):
        super().__init__()
        self.kernel = kernel
        self.components = nn.Parameter(components)
        self.normalization = nn.Parameter(normalization)

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [Nystroem]

    @classmethod
    def wrap(cls, obj: Nystroem) -> "TorchNystroem":
        kernel = Kernel.wrap(obj)
        return cls(
            kernel=kernel,
            components=torch.from_numpy(obj.components_),
            normalization=torch.from_numpy(obj.normalization_),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, self.components) @ self.normalization.t()
