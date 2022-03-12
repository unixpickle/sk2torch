import numpy as np
import pytest
import torch
import torch.jit
from sklearn.svm import SVR

from .svr import TorchSVR


@pytest.mark.parametrize(
    ("kernel",),
    [
        ("linear",),
        ("rbf",),
    ],
)
def test_svr(kernel):
    x = np.random.random(size=(100, 2)) * 2 - 1
    y = np.sin(x[:, 0]) + np.cos(x[:, 1]) * 0.2

    sk_obj = SVR(kernel=kernel)
    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchSVR.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.predict(x)
        actual = th_obj.predict(x_th).numpy()
        assert (np.abs(expected - actual) < 1e-5).all()
