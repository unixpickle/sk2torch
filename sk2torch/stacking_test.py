import warnings

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, LinearSVR

from .stacking import TorchStackingClassifier, TorchStackingRegressor


@pytest.mark.parametrize(
    ("drop_classifier", "binary", "passthrough", "method"),
    [
        (drop_classifier, binary, passthrough, method)
        for drop_classifier in [False, True]
        for binary in [False, True]
        for passthrough in [False, True]
        for method in ["auto", "predict_proba", "decision_function", "predict"]
        if not passthrough or (method == "auto" and binary == False)
    ],
)
def test_stacking_classifier(
    drop_classifier: bool, binary: bool, passthrough: bool, method: str
):
    rs = np.random.RandomState(1338)
    xs = rs.normal(size=(1000, 10))
    noised_xs = xs + rs.normal(size=xs.shape) * 0.1
    if binary:
        ys = (np.mean(noised_xs, axis=-1) > 0).astype(np.int32) * 2 + 1
    else:
        float_val = np.sum(noised_xs, axis=-1) / np.sqrt(xs.shape[1])
        ys = np.round(np.clip(float_val, -3, 3)).astype(np.int32) * 2

    classifiers = [
        ("dummy", DummyClassifier(strategy="prior")),  # no decision_function
        ("sgd_classifier", SGDClassifier(loss="log")),  # all methods supported
        ("linear_svc", LinearSVC()),  # no predict_proba
    ]
    if method == "decision_function":
        del classifiers[0]
    elif method == "predict_proba":
        del classifiers[2]

    sk_obj = StackingClassifier(
        estimators=classifiers,
        stack_method=method,
        passthrough=passthrough,
    )
    if drop_classifier:
        sk_obj.set_params(sgd_classifier="drop")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        sk_obj.fit(xs, ys)

    th_obj = torch.jit.script(TorchStackingClassifier.wrap(sk_obj))

    xs_th = torch.from_numpy(xs)
    with torch.no_grad():
        actual = th_obj.transform(xs_th).numpy()
        expected = sk_obj.transform(xs)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

        actual = th_obj.predict(xs_th).numpy()
        expected = sk_obj.predict(xs)
        assert actual.shape == expected.shape
        assert (actual == expected).all()

        actual = th_obj.decision_function(xs_th).numpy()
        expected = sk_obj.decision_function(xs)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

        actual = th_obj.predict_proba(xs_th).numpy()
        expected = sk_obj.predict_proba(xs)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    ("drop_model", "passthrough"),
    [
        (drop_model, passthrough)
        for drop_model in [False, True]
        for passthrough in [False, True]
    ],
)
def test_stacking_regressor(drop_model: bool, passthrough: bool):
    rs = np.random.RandomState(1338)
    xs = rs.normal(size=(1000, 10))
    ys = np.mean((xs + rs.normal(size=xs.shape) * 0.1 + 0.1) ** 2, axis=-1)

    classifiers = [
        ("dummy", DummyRegressor()),
        ("linear_svr", LinearSVR()),
    ]

    sk_obj = StackingRegressor(
        estimators=classifiers,
        passthrough=passthrough,
    )
    if drop_model:
        sk_obj.set_params(linear_svr="drop")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        sk_obj.fit(xs, ys)

    th_obj = torch.jit.script(TorchStackingRegressor.wrap(sk_obj))

    xs_th = torch.from_numpy(xs)
    with torch.no_grad():
        actual = th_obj.transform(xs_th).numpy()
        expected = sk_obj.transform(xs)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

        actual = th_obj.predict(xs_th).numpy()
        expected = sk_obj.predict(xs)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)
