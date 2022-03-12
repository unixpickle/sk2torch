import warnings
from typing import Callable, Tuple

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .logistic_regression import TorchLogisticRegression
from .sgd_classifier_test import xor_and_dataset, xor_dataset


@pytest.mark.parametrize(
    ("dataset", "fit_intercept", "space_classes", "multi_class"),
    [
        (load_breast_cancer, True, False, "auto"),
        (load_breast_cancer, True, True, "auto"),
        (load_digits, True, False, "auto"),
        (load_digits, True, True, "auto"),
        (xor_dataset, True, False, "auto"),
        (xor_dataset, True, False, "multinomial"),
        (xor_dataset, True, False, "ovr"),
        (xor_and_dataset, True, False, "auto"),
        (xor_and_dataset, True, False, "ovr"),
        (xor_and_dataset, True, False, "multinomial"),
        (xor_and_dataset, False, False, "auto"),
    ],
)
def test_sgd_classifier(
    dataset: Callable[..., Tuple[np.ndarray, np.ndarray]],
    fit_intercept: bool,
    space_classes: bool,
    multi_class: str,
):
    x, y = dataset(return_X_y=True)
    if space_classes:
        n_classes = np.max(y)
        y = np.where(y > n_classes // 2, 1 + (n_classes - y) + n_classes // 2, y)

    sk_obj = LogisticRegression(fit_intercept=fit_intercept, multi_class=multi_class)
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

        expected = sk_obj.predict_log_proba(x)
        actual = th_obj.predict_log_proba(x_th).numpy()
        assert actual.shape == expected.shape
        assert not np.isnan(expected).any()
        assert np.allclose(np.exp(expected), np.exp(actual))

        expected = sk_obj.predict_proba(x)
        actual = th_obj.predict_proba(x_th).numpy()
        assert actual.shape == expected.shape
        assert np.isfinite(expected).any()
        assert np.isfinite(actual).any()
        assert np.allclose(expected, actual)
