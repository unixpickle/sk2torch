from typing import List, Type

import torch
import torch.nn as nn
from sklearn.compose import TransformedTargetRegressor


class TorchTransformedTargetRegressor(nn.Module):
    def __init__(self, regressor: nn.Module, transformer: nn.Module, training_dim: int):
        super().__init__()
        self.regressor = regressor
        self.transformer = transformer
        self.training_dim = training_dim

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [TransformedTargetRegressor]

    @classmethod
    def wrap(cls, obj: TransformedTargetRegressor) -> "TorchTransformedTargetRegressor":
        assert (
            obj.transformer_ is not None
        ), "identity and function transformers not supported"

        from .wrap import wrap

        return TorchTransformedTargetRegressor(
            regressor=wrap(obj.regressor_),
            transformer=wrap(obj.transformer_),
            training_dim=obj._training_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self.regressor(x)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = self.transformer.inverse_transform(y)
        if self.training_dim == 1 and len(y.shape) == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        return y
