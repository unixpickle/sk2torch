import numpy as np
import pytest
import torch
import torch.jit
from sklearn.kernel_approximation import Nystroem

from .kernel_test import create_test_data
from .nystroem import TorchNystroem


@pytest.mark.parametrize(
    ("kernel", "gamma", "coef0", "degree"),
    [
        ("rbf", None, None, None),
        ("sigmoid", None, None, None),
        ("polynomial", None, None, None),
        ("linear", None, None, None),
        ("rbf", 0.5, None, None),
        ("sigmoid", 0.5, -0.2, None),
        ("polynomial", 0.5, -0.2, 2),
    ],
)
def test_nystroem_defaults(kernel, gamma, coef0, degree):
    x_th, _ = create_test_data()
    x = x_th.numpy()
    sk_obj = Nystroem(
        n_components=5, kernel=kernel, gamma=gamma, coef0=coef0, degree=degree
    )
    sk_obj.fit(x)
    th_obj = torch.jit.script(TorchNystroem.wrap(sk_obj))

    with torch.no_grad():
        expected = sk_obj.transform(x)
        actual = th_obj.transform(x_th).numpy()
        assert (np.abs(actual - expected) < 1e-5).all()
