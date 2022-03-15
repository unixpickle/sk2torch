import numpy as np
import torch
import torch.jit
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .linear_model_test import xor_and_dataset
from .pipeline import TorchPipeline


def test_pipeline_classifier():
    x, y = xor_and_dataset()
    sk_obj = Pipeline(
        [("scale", StandardScaler()), ("classify", SGDClassifier(loss="log"))]
    )
    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchPipeline.wrap(sk_obj))

    x_th = torch.from_numpy(x).clone()
    with torch.no_grad():
        expected = sk_obj.predict(x)
        actual = th_obj.predict(x_th).numpy()
        assert (expected == actual).all()

        # forward() should be equivalent
        actual = th_obj(x_th).numpy()
        assert (expected == actual).all()

        expected = sk_obj.decision_function(x)
        actual = th_obj.decision_function(x_th).numpy()
        assert np.allclose(expected, actual)

        expected = sk_obj.predict_proba(x)
        actual = th_obj.predict_proba(x_th).numpy()
        assert np.allclose(expected, actual)

        expected = sk_obj.predict_log_proba(x)
        actual = th_obj.predict_log_proba(x_th).numpy()
        assert np.allclose(expected, actual)


def test_pipeline_transform():
    x, y = xor_and_dataset()
    sk_obj = Pipeline(
        [
            ("center", StandardScaler(with_std=False)),
            ("scale", StandardScaler()),
        ]
    )
    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchPipeline.wrap(sk_obj))

    x_th = torch.from_numpy(x)
    with torch.no_grad():
        expected = sk_obj.transform(x)
        actual = th_obj.transform(x_th).numpy()
        assert np.allclose(expected, actual)

        # forward() should be equivalent
        actual = th_obj(x_th).numpy()
        assert np.allclose(expected, actual)

        expected = sk_obj.inverse_transform(x)
        actual = th_obj.inverse_transform(x_th).numpy()
        assert np.allclose(expected, actual)


def test_pipeline_transform_no_inverse():
    x, y = xor_and_dataset()
    sk_obj = Pipeline(
        [
            ("center", Nystroem(n_components=2)),
            ("scale", StandardScaler()),
        ]
    )
    sk_obj.fit(x, y)
    th_obj = torch.jit.script(TorchPipeline.wrap(sk_obj))

    x_th = torch.from_numpy(x)
    with torch.no_grad():
        expected = sk_obj.transform(x)
        actual = th_obj.transform(x_th).numpy()
        assert np.allclose(expected, actual)
