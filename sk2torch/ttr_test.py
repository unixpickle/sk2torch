import numpy as np
import torch
import torch.jit
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .ttr import TorchTransformedTargetRegressor


def test_ttr():
    x = np.random.random(size=(100, 2)) * 2 - 1
    y = 10000 * (np.sin(x[:, 0]) + np.cos(x[:, 1]) * 0.2) - 20000

    sk_obj = TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler())
    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchTransformedTargetRegressor.wrap(sk_obj))

    with torch.no_grad():
        x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.predict(x)
        actual = th_obj.predict(x_th).numpy()
        assert (np.abs(expected - actual) < 0.1).all()
