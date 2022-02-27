from typing import Tuple

import numpy as np
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import SGDClassifier

from .sgd_classifier import TorchSGDClassifier


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
    ("dataset", "loss", "check_probs"),
    [
        (load_breast_cancer, "log", False),
        (load_digits, "log", False),
        (xor_dataset, "log", True),
        (xor_and_dataset, "log", True),
    ],
)
def test_linear_sgd(dataset, loss, check_probs):
    x, y = dataset(return_X_y=True)
    sk_obj = SGDClassifier(loss=loss)
    sk_obj.fit(x, y)
    th_obj = TorchSGDClassifier.wrap(sk_obj)

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
