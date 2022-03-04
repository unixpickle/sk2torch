import itertools

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.tree import DecisionTreeRegressor

from .tree import TorchDecisionTreeRegressor


@pytest.mark.parametrize(
    ("target", "max_depth"),
    list(itertools.product(["1d", "1d-nonflat", "2d"], [1, 3, None])),
)
def test_decision_tree_regressor(target: str, max_depth: int):
    xs = np.random.RandomState(1337).normal(size=(1000, 2))
    norms = np.sum(xs ** 2, axis=-1)
    if target == "2d":
        ys = np.stack([norms, np.sqrt(norms)], axis=1)
    else:
        if target == "1d":
            ys = norms
        else:
            ys = norms[:, None]

    sk_obj = DecisionTreeRegressor(max_depth=max_depth)
    sk_obj.fit(xs, ys)
    th_obj = torch.jit.script(TorchDecisionTreeRegressor.wrap(sk_obj))

    xs_th = torch.from_numpy(xs)

    with torch.no_grad():
        actual = th_obj.predict(xs_th).numpy()
        expected = sk_obj.predict(xs)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

        actual = th_obj.decision_path(xs_th).numpy()
        expected = sk_obj.decision_path(xs)
        assert (actual == expected).all()
