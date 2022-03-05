import itertools

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .tree import TorchDecisionTreeClassifier, TorchDecisionTreeRegressor


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


@pytest.mark.parametrize(
    ("target", "max_depth"),
    list(itertools.product(["binary", "trinary", "many"], [1, 3, None])),
)
def test_decision_tree_classifier(target: str, max_depth: int):
    xs = np.random.RandomState(1337).normal(size=(1000, 3))
    norms = np.sqrt(np.mean(xs ** 2, axis=-1))
    if target == "binary":
        ys = norms > np.mean(norms)
    elif target == "trinary":
        ys = (
            (norms > np.mean(norms)).astype(np.int32)
            + (norms > np.mean(norms) + 0.1).astype(np.int32)
            + 13
        )
    else:
        # One label with three classes, the other with two.
        y1 = (
            (norms > np.mean(norms)).astype(np.int32)
            + (norms > np.mean(norms) + 0.1).astype(np.int32)
            + 1
        )
        y2 = (norms < np.mean(norms) - 0.1).astype(np.int32) + 3
        ys = np.stack([y1, y2], axis=1)

    sk_obj = DecisionTreeClassifier(max_depth=max_depth)
    sk_obj.fit(xs, ys)
    th_obj = torch.jit.script(TorchDecisionTreeClassifier.wrap(sk_obj))

    xs_th = torch.from_numpy(xs)

    with torch.no_grad():
        actual = th_obj.predict(xs_th).numpy()
        expected = sk_obj.predict(xs)
        assert actual.shape == expected.shape
        assert (actual == expected).all()

        actual = th_obj.decision_path(xs_th).numpy()
        expected = sk_obj.decision_path(xs)
        assert (actual == expected).all()

        actual = th_obj.predict_proba(xs_th)
        expected = sk_obj.predict_proba(xs)
        assert isinstance(actual, list) == (len(ys.shape) > 1)
        assert len(actual) == len(expected)
        for a, x in zip(actual, expected):
            a = a.numpy()
            assert a.shape == x.shape
            assert np.allclose(a, x)

        actual = th_obj.predict_log_proba(xs_th)
        with np.errstate(divide="ignore"):  # ignore log of 0
            expected = sk_obj.predict_log_proba(xs)
        assert isinstance(actual, list) == (len(ys.shape) > 1)
        assert len(actual) == len(expected)
        for a, x in zip(actual, expected):
            a = a.numpy()
            assert a.shape == x.shape
            assert np.allclose(np.exp(a), np.exp(x))
