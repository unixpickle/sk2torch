import itertools
from collections import Counter

import numpy as np
import pytest
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor

from .dummy import TorchDummyClassifier, TorchDummyRegressor


@pytest.mark.parametrize(
    ("strategy", "multi_class"),
    list(
        itertools.product(
            ["most_frequent", "prior", "stratified", "uniform", "constant"],
            [False, True],
        )
    ),
)
def test_dummy_classifier(strategy: str, multi_class: bool):
    rs = np.random.RandomState(1337)
    xs = rs.normal(size=(10000, 3))
    if multi_class:
        ys = np.concatenate(
            [
                rs.randint(low=1, high=3, size=(xs.shape[0], 1)),
                rs.randint(low=-1, high=5, size=(xs.shape[0], 1)),
            ],
            axis=1,
        )
        constant = np.array([2, 3])
    else:
        ys = rs.choice(
            a=[-2, 1, 3, 5, 7, 6],
            p=[1 / 6, 1 / 6 - 1 / 12, 1 / 6 + 1 / 12, 1 / 6, 1 / 6, 1 / 6],
            size=(xs.shape[0],),
        )
        constant = 1

    sk_obj = DummyClassifier(
        strategy=strategy, constant=constant if strategy == "constant" else None
    )
    sk_obj.fit(xs, ys)
    th_obj = TorchDummyClassifier.wrap(sk_obj)

    xs_th = torch.from_numpy(xs)

    with torch.no_grad():
        expected = sk_obj.predict(xs)
        actual = th_obj.predict(xs_th).numpy()
        assert expected.shape == actual.shape
        if strategy in ["most_frequent", "prior", "constant"]:
            assert (actual == expected).all()
        else:
            # Make sure the sampling probabilities are the same, even though
            # the outputs are non-deterministic.
            threshold = 0.05
            if len(expected.shape) == 1:
                actual, expected = actual[:, None], expected[:, None]
            for i in range(expected.shape[1]):
                x, a = expected[:, i], actual[:, i]
                assert x.shape == a.shape
                x_counts = Counter(x)
                a_counts = Counter(a)
                assert set(x_counts.keys()) == set(a_counts.keys())
                for k, x_count in x_counts.items():
                    a_count = a_counts[k]
                    assert abs(x_count / len(x) - a_count / len(a)) < threshold

        expected = sk_obj.predict_proba(xs)
        actual = th_obj.predict_proba(xs_th)
        assert len(expected) == len(actual)
        if strategy in ["most_frequent", "prior", "uniform", "constant"]:
            for x, a in zip(expected, actual):
                assert x.shape == a.shape
                assert np.allclose(x, a.numpy())
        else:
            # Make sure the sampling probabilities are the same, even though
            # the outputs are non-deterministic.
            threshold = 0.05
            if not isinstance(expected, list):
                expected, actual = [expected], [actual]
            for x, a in zip(expected, actual):
                a = a.numpy()
                assert not np.allclose(x, a)
                assert x.shape == a.shape
                assert (
                    np.max(np.abs(np.mean(x, axis=0) - np.mean(a, axis=0))) < threshold
                )


@pytest.mark.parametrize(
    ("strategy", "multi_class"),
    list(
        itertools.product(
            ["mean", "median", "quantile", "constant"],
            [False, True],
        )
    ),
)
def test_dummy_regressor(strategy: str, multi_class: bool):
    xs = np.random.normal(size=(10000, 3))
    ys = np.random.normal(size=(10000,) if not multi_class else (10000, 3))

    sk_obj = DummyRegressor(
        strategy=strategy,
        constant=ys[0] if strategy == "constant" else None,
        quantile=0.5 if strategy == "quantile" else None,
    )
    sk_obj.fit(xs, ys)
    th_obj = TorchDummyRegressor.wrap(sk_obj)

    xs_th = torch.from_numpy(xs)

    with torch.no_grad():
        expected = sk_obj.predict(xs)
        actual = th_obj.predict(xs_th).numpy()
        assert expected.shape == actual.shape
        assert np.allclose(expected, actual)
