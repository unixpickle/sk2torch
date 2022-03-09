import warnings
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .nn import TorchMLPClassifier, TorchMLPRegressor


@pytest.mark.parametrize(
    ("activation", "class_type", "hidden_sizes"),
    [
        ("relu", "binary", (15, 20)),
        ("relu", "multiclass", (15, 20)),
        ("relu", "multilabel-indicator", (15, 20)),
        ("tanh", "multiclass", (15, 20)),
        ("logistic", "multiclass", (15, 20)),
        ("identity", "multiclass", (15, 20)),
        ("tanh", "multiclass", (15)),
        ("tanh", "multiclass", (30, 30, 30)),
    ],
)
def test_mlp_classifier(
    activation: str, class_type: str, hidden_sizes: Tuple[int, ...]
):
    rng = np.random.RandomState(1337)
    x = rng.normal(size=(1000, 10))
    perturbed = x + rng.normal(size=x.shape)
    if class_type == "binary":
        y = perturbed.sum(1) > 0
    elif class_type == "multiclass":
        y = perturbed.argmax(1)
    elif class_type == "multilabel-indicator":
        y = perturbed > 0

    sk_model = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation=activation,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        sk_model.fit(x[:-100], y[:-100])
    th_model = torch.jit.script(TorchMLPClassifier.wrap(sk_model))

    x = x[-100:]
    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_model.predict_proba(x)
        actual = th_model.predict_proba(x_th).numpy()
        assert expected.shape == actual.shape
        assert (np.abs(actual - expected) < 1e-3).all()

        expected = sk_model.predict(x)
        actual = th_model.predict(x_th).numpy()
        assert expected.shape == actual.shape
        assert (actual == expected).all()


@pytest.mark.parametrize(
    ("activation", "output_2d", "hidden_sizes"),
    [
        ("relu", False, (15, 20)),
        ("tanh", False, (15, 20)),
        ("logistic", False, (15, 20)),
        ("identity", False, (15, 20)),
        ("relu", True, (15, 20)),
    ],
)
def test_mlp_regressor(activation: str, output_2d: bool, hidden_sizes: Tuple[int, ...]):
    rng = np.random.RandomState(1337)
    x = rng.normal(size=(1000, 10))
    perturbed = x + rng.normal(size=x.shape)
    if not output_2d:
        y = perturbed.sum(1)
    else:
        y = np.stack([perturbed.sum(1), np.sqrt(np.abs(perturbed.sum(1)))], axis=-1)

    sk_model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation=activation,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        sk_model.fit(x[:-100], y[:-100])
    th_model = torch.jit.script(TorchMLPRegressor.wrap(sk_model))

    x = x[-100:]
    x_th = torch.from_numpy(x)

    with torch.no_grad():
        expected = sk_model.predict(x)
        actual = th_model.predict(x_th).numpy()
        assert expected.shape == actual.shape
        assert np.allclose(actual, expected)
