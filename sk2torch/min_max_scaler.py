from typing import List, Tuple, Type, Union

import numpy as np
import torch
import torch.jit
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class TorchMinMaxScaler(nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        min_: torch.Tensor,
        feature_range: torch.Tensor,
        clip: bool,
    ):
        super().__init__()
        self.scale = nn.Parameter(scale)
        self.min = nn.Parameter(min_)
        self.register_buffer("feature_range", feature_range)
        self.clip = clip

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [MinMaxScaler]

    @classmethod
    def wrap(cls, obj: MinMaxScaler) -> "TorchMinMaxScaler":
        scale = torch.from_numpy(obj.scale_)
        min_ = torch.from_numpy(obj.min_)
        return TorchMinMaxScaler(
            scale,
            min_,
            torch.tensor(obj.feature_range).to(scale),
            obj.clip,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.min
        if self.clip:
            x = torch.minimum(
                torch.maximum(x, self.feature_range[0]), self.feature_range[1]
            )
        return x

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min) / self.scale
