import numpy as np
import pytest
import torch
import torch.jit
from sklearn.preprocessing import StandardScaler

from .standard_scaler import TorchStandardScaler


@pytest.mark.parametrize(
    ("with_mean", "with_std"), [(True, True), (True, False), (False, True)]
)
def test_standard_scaler(with_mean, with_std):
    x = np.random.normal(size=(30, 3))
    x *= np.array([2.0, 0.3, 1.0])
    x += np.array([1.0, 2.0, 3.0])
    sk_obj = StandardScaler(with_mean=with_mean, with_std=with_std)
    sk_obj.fit(x)
    th_obj = torch.jit.script(TorchStandardScaler.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.transform(x, copy=True)
        actual = th_obj(x_th).numpy()
        assert np.allclose(expected, actual)

        actual = th_obj.inverse_transform(torch.from_numpy(expected)).numpy()
        expected = sk_obj.inverse_transform(expected, copy=True)
        assert np.allclose(expected, actual)
