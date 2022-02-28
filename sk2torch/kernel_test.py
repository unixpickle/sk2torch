from typing import Optional

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.metrics.pairwise import pairwise_kernels

from .kernel import Kernel


@pytest.mark.parametrize(
    ("metric", "gamma", "coef0", "degree"),
    [
        ("linear", None, None, None),
        ("rbf", 0.3, None, None),
        ("poly", 0.3, 1.3, 2),
        ("sigmoid", 0.3, 1.5, None),
    ],
)
def test_kernel(
    metric: str, gamma: Optional[float], coef0: Optional[float], degree: Optional[float]
):
    torch.manual_seed(0)
    xs = torch.randn(15, 5)
    ys = torch.randn(13, 5)
    kernel = torch.jit.script(
        Kernel(metric=metric, gamma=gamma, coef0=coef0, degree=degree)
    )
    actual = kernel(xs, ys).numpy()
    args = {
        k: v
        for k, v in [("gamma", gamma), ("coef0", coef0), ("degree", degree)]
        if v is not None
    }
    expected = pairwise_kernels(xs.numpy(), ys.numpy(), metric=metric, **args)
    assert (np.abs(actual - expected) < 1e-5).all()
