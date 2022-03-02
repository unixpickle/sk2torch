from typing import Optional, Tuple

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
        ("rbf", None, None, None),
        ("rbf", 0.3, None, None),
        ("poly", None, None, None),
        ("poly", 0.3, 1.3, 2),
        ("sigmoid", None, None, None),
        ("sigmoid", 0.3, 1.5, None),
    ],
)
def test_kernel(
    metric: str, gamma: Optional[float], coef0: Optional[float], degree: Optional[float]
):
    xs, ys = create_test_data()
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


def create_test_data() -> Tuple[torch.Tensor, torch.Tensor]:
    t1 = torch.tensor(
        [
            -1.1258,
            -1.1524,
            -0.2506,
            -0.4339,
            0.8487,
            0.6920,
            -0.3160,
            -2.1152,
            0.3223,
            -1.2633,
            0.3500,
            0.3081,
            0.1198,
            1.2377,
            1.1168,
            -0.2473,
            -1.3527,
            -1.6959,
            0.5667,
            0.7935,
            0.5988,
            -1.5551,
            -0.3414,
            1.8530,
            0.7502,
            -0.5855,
            -0.1734,
            0.1835,
            1.3894,
            -0.6787,
            0.9383,
            0.4889,
            -0.6731,
            0.8728,
            1.0554,
            0.1778,
            -0.2303,
            -0.3067,
            -1.5810,
            1.7066,
            -0.4462,
            0.7440,
            1.5210,
            3.4105,
            -1.5312,
        ],
        dtype=torch.float64,
    )
    t2 = torch.tensor(
        [
            0.9625,
            0.3492,
            -0.9215,
            -0.0562,
            -0.7015,
            1.0367,
            -0.6037,
            -1.2788,
            0.1239,
            1.1648,
            0.9234,
            1.3873,
            1.3750,
            0.6596,
            0.4766,
            -1.0163,
            0.6104,
            0.4669,
            1.9507,
            -1.0631,
            1.1404,
            -0.0899,
            0.7298,
            0.1723,
            -1.6115,
            -0.4794,
            -0.1434,
            -0.3173,
            0.9671,
            -0.9911,
            0.3016,
            0.0788,
            0.8629,
            -0.0195,
            0.9910,
            -0.7777,
            0.3140,
            0.2133,
            -0.1201,
        ],
        dtype=torch.float64,
    )
    return t1.view(-1, 3), t2.view(-1, 3)
