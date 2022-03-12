import warnings
from typing import Callable, Tuple, Union

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.svm import LinearSVC, LinearSVR

from .linear_model import (
    TorchLinearClassifier,
    TorchLinearRegression,
    TorchLogisticRegression,
    TorchSGDClassifier,
)
from .svc_test import quadrant_dataset


def noisy_binary_mean(**_):
    """A dataset with a simple linear decision function but noisy labels"""
    xs = np.random.normal(size=(1000, 10))
    ys = np.mean(xs + 0.1 + np.random.normal(size=xs.shape) * 0.1, axis=-1) < 0
    return xs, ys.astype(np.int32)


def xor_dataset(**_) -> Tuple[np.ndarray, np.ndarray]:
    """
    A binary classification problem that is not linearly separable.
    """
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    return np.tile(x, [10, 1]), np.tile(y, [10])


def xor_and_dataset(**_) -> Tuple[np.ndarray, np.ndarray]:
    """
    A three-class problem that is not linearly separable.
    """
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 2])
    return np.tile(x, [10, 1]), np.tile(y, [10])


@pytest.mark.parametrize(
    (
        "sk_obj",
        "num_targets",
    ),
    [
        (cls(fit_intercept=fit_intercept), num_targets)
        for cls in [LinearRegression, Ridge, RidgeCV]
        for fit_intercept in [False, True]
        for num_targets in [1, 2]
    ]
    + [(SGDRegressor(fit_intercept=False), 1), (SGDRegressor(fit_intercept=True), 1)],
)
def test_linear_regression(
    sk_obj: Union[LinearRegression, Ridge, RidgeCV], num_targets: int
):
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

    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchLinearRegression.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.predict(x)
        actual = th_obj.predict(x_th).numpy()
        assert actual.shape == expected.shape
        assert (np.abs(expected - actual) < 1e-5).all()


@pytest.mark.parametrize(
    ("fit_intercept",),
    [
        (False,),
        (True,),
    ],
)
def test_linear_svr(fit_intercept: bool):
    x = np.random.random(size=(100, 2)) * 2 - 1
    y = np.sin(x[:, 0]) + np.cos(x[:, 1]) * 0.2

    sk_obj = LinearSVR(
        fit_intercept=fit_intercept, intercept_scaling=2.314, max_iter=10000
    )
    sk_obj.fit(x, y)
    assert LinearSVR in TorchLinearRegression.supported_classes()
    th_obj = torch.jit.script(TorchLinearRegression.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.predict(x)
        actual = th_obj.predict(x_th).numpy()
        assert (np.abs(expected - actual) < 1e-5).all()


@pytest.mark.parametrize(
    ("dataset", "loss", "check_probs", "fit_intercept", "space_classes"),
    [
        (load_breast_cancer, "log", False, True, False),
        (load_breast_cancer, "log", False, True, True),
        (load_digits, "log", False, True, False),
        (load_digits, "log", False, True, True),
        (xor_dataset, "log", True, True, False),
        (xor_and_dataset, "log", True, True, False),
        (xor_and_dataset, "log", True, False, False),
        (xor_and_dataset, "hinge", False, True, False),
    ],
)
def test_sgd_classifier(
    dataset: Callable[..., Tuple[np.ndarray, np.ndarray]],
    loss: str,
    check_probs: bool,
    fit_intercept: bool,
    space_classes: bool,
):
    x, y = dataset(return_X_y=True)
    if space_classes:
        n_classes = np.max(y)
        y = np.where(y > n_classes // 2, 1 + (n_classes - y) + n_classes // 2, y)

    sk_obj = SGDClassifier(loss=loss, fit_intercept=fit_intercept)
    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchSGDClassifier.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.decision_function(x)
        actual = th_obj.decision_function(x_th).numpy()
        assert np.allclose(expected, actual)

        expected = sk_obj.predict(x)
        actual = th_obj(x_th).numpy()
        assert (expected == actual).all()

        if check_probs:
            expected = sk_obj.predict_log_proba(x)
            actual = th_obj.predict_log_proba(x_th).numpy()
            assert not np.isnan(expected).any()
            assert np.allclose(np.exp(expected), np.exp(actual))

            expected = sk_obj.predict_proba(x)
            actual = th_obj.predict_proba(x_th).numpy()
            assert np.isfinite(expected).any()
            assert np.isfinite(actual).any()
            assert np.allclose(expected, actual)


@pytest.mark.parametrize(
    ("dataset", "fit_intercept", "space_classes", "multi_class", "cv"),
    [
        (load_breast_cancer, True, False, "auto", False),
        (load_breast_cancer, True, True, "auto", False),
        (load_digits, True, False, "auto", False),
        (load_digits, True, True, "auto", False),
        (xor_dataset, True, False, "auto", False),
        (xor_dataset, True, False, "multinomial", False),
        (xor_dataset, True, False, "ovr", False),
        (xor_and_dataset, True, False, "auto", False),
        (xor_and_dataset, True, False, "ovr", False),
        (xor_and_dataset, True, False, "multinomial", False),
        (xor_and_dataset, False, False, "auto", False),
        (noisy_binary_mean, True, True, "ovr", False),
        (noisy_binary_mean, True, True, "multinomial", False),
        (noisy_binary_mean, True, True, "auto", True),
    ],
)
def test_logistic_regression(
    dataset: Callable[..., Tuple[np.ndarray, np.ndarray]],
    fit_intercept: bool,
    space_classes: bool,
    multi_class: str,
    cv: bool,
):
    x, y = dataset(return_X_y=True)
    if space_classes:
        n_classes = np.max(y)
        y = np.where(y > n_classes // 2, 1 + (n_classes - y) + n_classes // 2, y)

    sk_obj = (LogisticRegression if not cv else LogisticRegressionCV)(
        fit_intercept=fit_intercept, multi_class=multi_class
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchLogisticRegression.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.decision_function(x)
        actual = th_obj.decision_function(x_th).numpy()
        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)

        expected = sk_obj.predict(x)
        actual = th_obj(x_th).numpy()
        assert actual.shape == expected.shape
        assert (expected == actual).all()

        expected = sk_obj.predict_proba(x)
        actual = th_obj.predict_proba(x_th).numpy()
        assert actual.shape == expected.shape
        assert np.isfinite(expected).any()
        assert np.isfinite(actual).any()
        assert np.allclose(expected, actual)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            expected = sk_obj.predict_log_proba(x)
        actual = th_obj.predict_log_proba(x_th).numpy()
        assert actual.shape == expected.shape
        assert not np.isnan(expected).any()
        assert np.allclose(np.exp(expected), np.exp(actual))


@pytest.mark.parametrize(
    ("dataset", "fit_intercept", "space_classes", "cv"),
    [
        (load_breast_cancer, True, False, False),
        (load_breast_cancer, True, True, False),
        (load_breast_cancer, True, True, True),
        (load_breast_cancer, False, True, True),
        (load_digits, True, False, False),
        (load_digits, True, True, False),
        (load_digits, True, True, True),
        (load_digits, False, True, True),
    ],
)
def test_ridge_classifier(
    dataset: Callable[..., Tuple[np.ndarray, np.ndarray]],
    fit_intercept: bool,
    space_classes: bool,
    cv: bool,
):
    x, y = dataset(return_X_y=True)
    if space_classes:
        n_classes = np.max(y)
        y = np.where(y > n_classes // 2, 1 + (n_classes - y) + n_classes // 2, y)

    sk_obj = (RidgeClassifier if not cv else RidgeClassifierCV)(
        fit_intercept=fit_intercept
    )
    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchLinearClassifier.wrap(sk_obj))

    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_obj.decision_function(x)
        actual = th_obj.decision_function(x_th).numpy()
        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)

        expected = sk_obj.predict(x)
        actual = th_obj(x_th).numpy()
        assert actual.shape == expected.shape
        assert (expected == actual).all()


@pytest.mark.parametrize(
    ("fit_intercept", "n_classes", "space_classes"),
    [
        (False, 2, False),
        (True, 2, False),
        (False, 3, False),
        (True, 3, False),
        (True, 4, True),
    ],
)
def test_linear_svc(fit_intercept: bool, n_classes: int, space_classes: int):
    xs, ys = quadrant_dataset(n_classes, space_classes)

    test_xs = np.random.random(size=(128, 2)) * 2 - 1
    test_xs_th = torch.from_numpy(test_xs)

    model = LinearSVC(fit_intercept=fit_intercept, intercept_scaling=2.314)
    model.fit(xs, ys)
    model_th = torch.jit.script(TorchLinearClassifier.wrap(model))

    with torch.no_grad():
        expected = model.decision_function(test_xs)
        actual = model_th.decision_function(test_xs_th).numpy()
        assert (np.abs(actual - expected) < 1e-8).all()

        expected = model.predict(test_xs)
        actual = model_th.predict(test_xs_th).numpy()
        assert (actual == expected).all()
