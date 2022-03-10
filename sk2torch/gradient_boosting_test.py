import numpy as np
import pytest
import torch
import torch.jit
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from .gradient_boosting import (
    TorchGradientBoostingClassifier,
    TorchGradientBoostingRegressor,
)


@pytest.mark.parametrize(
    ("target_type", "loss", "init_zero"),
    [
        ("binary", "exponential", False),
        ("binary", "exponential", True),
        ("binary", "deviance", False),
        ("binary", "deviance", True),
        ("multi", "deviance", False),
        ("multi", "deviance", True),
    ],
)
def test_gradient_boosting_classifier(target_type, loss, init_zero):
    xs = np.random.normal(size=(1000, 3)) * 2
    noised = xs + np.random.normal(size=xs.shape) * 0.1
    if target_type == "binary":
        ys = np.mean(noised, axis=1) < 0
    else:
        # Non-uniformly spaced or centered class labels.
        ys = np.round(np.clip(np.mean(noised, axis=1), -3, 3)).astype(np.int32) * 2

    sk_obj = GradientBoostingClassifier(
        loss=loss, init="zero" if init_zero else None, n_estimators=5
    )
    sk_obj.fit(xs, ys)
    th_obj = torch.jit.script(TorchGradientBoostingClassifier.wrap(sk_obj))
    xs_th = torch.from_numpy(xs)

    with torch.no_grad():
        expected = sk_obj.decision_function(xs)
        actual = th_obj.decision_function(xs_th).numpy()
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

        expected = sk_obj.predict_proba(xs)
        actual = th_obj.predict_proba(xs_th).numpy()
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

        expected = sk_obj.predict(xs)
        actual = th_obj.predict(xs_th).numpy()
        assert actual.shape == expected.shape
        assert (actual == expected).all()


@pytest.mark.parametrize(("init_zero",), [(False,), (True,)])
def test_gradient_boosting_regressor(init_zero):
    xs = np.random.normal(size=(1000, 3)) * 2
    ys = np.mean(xs, axis=-1)
    sk_obj = GradientBoostingRegressor(
        init="zero" if init_zero else None, n_estimators=5
    )
    sk_obj.fit(xs, ys)
    th_obj = torch.jit.script(TorchGradientBoostingRegressor.wrap(sk_obj))
    xs_th = torch.from_numpy(xs)

    with torch.no_grad():
        expected = sk_obj.predict(xs)
        actual = th_obj.predict(xs_th).numpy()
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)
