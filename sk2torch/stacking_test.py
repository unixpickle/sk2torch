import warnings

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC

from .stacking import TorchStackingClassifier


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
        ("svc", SGDClassifier(loss="log")),  # all methods supported
        ("linear_svc", LinearSVC()),  # no predict_proba
    ]
    if method == "decision_function":
        del classifiers[0]
    elif method == "predict_proba":
        del classifiers[2]

    sk_obj = StackingClassifier(
        estimators=classifiers,
        final_estimator=SVC(probability=True),
        stack_method=method,
        passthrough=passthrough,
    )
    if drop_classifier:
        sk_obj.set_params(svc="drop")
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
        # Accuracy is lower because SVC predict_proba is slightly inaccurate
        assert (np.abs(actual - expected) < 3e-3).all()
