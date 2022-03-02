from copy import deepcopy
from typing import Union

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.svm import SVR, NuSVR

from .kernel import Kernel


class TorchSVR(nn.Module):
    def __init__(
        self,
        kernel: Kernel,
        support_vectors: torch.Tensor,
        dual_coef: torch.Tensor,
        intercept: torch.Tensor,
    ):
        super().__init__()
        self.kernel = kernel
        self.support_vectors = nn.Parameter(support_vectors)
        self.dual_coef = nn.Parameter(dual_coef)
        self.intercept = nn.Parameter(intercept)

    @classmethod
    def supports_wrap(cls, obj: BaseEstimator) -> bool:
        return isinstance(obj, SVR) or isinstance(obj, NuSVR)

    @classmethod
    def wrap(cls, obj: Union[SVR, NuSVR]) -> "TorchSVR":
        assert not obj._sparse, "sparse SVC not supported"
        return cls(
            kernel=Kernel.wrap(obj),
            support_vectors=torch.from_numpy(obj.support_vectors_),
            dual_coef=torch.from_numpy(obj.dual_coef_),
            intercept=torch.from_numpy(obj.intercept_),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict regression values for the given feature vectors.

        An alias for self.predict().
        """
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the given feature vectors.
        """
        kernel_out = self.kernel(x, self.support_vectors)
        return torch.einsum("jk,jk->j", self.dual_coef, kernel_out) + self.intercept
