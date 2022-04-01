from typing import Any

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.preprocessing import MinMaxScaler

from .min_max_scaler import TorchMinMaxScaler


@pytest.mark.parametrize(
    ("clip", "min", "max"),
    [
        (False, 0, 1),
        (False, -1, 0),
        (False, 1, 2),
        (False, -1, 1),
        (True, 0, 1),
        (True, -1, 1),
    ],
)
def test_min_max_scaler(clip: False, min: Any, max: Any):
    x = np.random.normal(size=(1000, 2))
    sk_obj = MinMaxScaler(feature_range=(min, max), clip=clip)
    sk_obj.fit(x)
    th_obj = torch.jit.script(TorchMinMaxScaler.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.transform(x)
        actual = th_obj.transform(x_th).numpy()
        assert np.allclose(actual, expected)

        y = expected.copy()
        expected = sk_obj.inverse_transform(y)
        actual = th_obj.inverse_transform(torch.from_numpy(y)).numpy()
        assert np.allclose(actual, expected)
