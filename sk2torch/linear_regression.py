from typing import List, Type

import numpy as np
import torch
import torch.jit
import torch.nn as nn
from sklearn.linear_model import LinearRegression


class TorchLinearRegression(nn.Module):
    def __init__(
        self,
        weights: torch.Tensor,
        biases: torch.Tensor,
    ):
        super().__init__()
        if len(weights.shape) == 1:
            weights = weights[None]
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [LinearRegression]

    @classmethod
    def wrap(cls, obj: LinearRegression) -> "TorchLinearRegression":
        return cls(
            weights=torch.from_numpy(obj.coef_),
            biases=torch.from_numpy(np.array(obj.intercept_)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the given feature vectors.

        An alias for self.predict().
        """
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the given feature vectors.
        """
        outputs = (x @ self.weights.t()) + self.biases
        if outputs.shape[1] == 1:
            return outputs.view(-1)
        return outputs
