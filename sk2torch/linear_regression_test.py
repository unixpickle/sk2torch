import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression

from .linear_regression import TorchLinearRegression


@pytest.mark.parametrize(
    (
        "num_targets",
        "fit_intercept",
    ),
    [
        (1, False),
        (1, True),
        (2, False),
        (2, True),
    ],
)
def test_linear_svr(num_targets: int, fit_intercept: bool):
    x = np.random.random(size=(100, 2)) * 2 - 1
    if num_targets == 1:
        y = np.sin(x[:, 0]) + np.cos(x[:, 1]) * 0.2
    else:
        assert num_targets == 2
        y = np.stack(
            [
                np.sin(x[:, 0]) + np.cos(x[:, 1]) * 0.2,
                np.cos(x[:, 0]) + np.sin(x[:, 1]) * 0.2,
            ],
            axis=1,
        )

    sk_obj = LinearRegression(fit_intercept=fit_intercept)
    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchLinearRegression.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.predict(x)
        actual = th_obj.predict(x_th).numpy()
        assert actual.shape == expected.shape
        assert (np.abs(expected - actual) < 1e-5).all()
