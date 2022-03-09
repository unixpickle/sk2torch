from typing import Any, List

import torch.nn as nn
from sklearn.base import BaseEstimator

from .dummy import TorchDummyClassifier
from .gradient_boosting import TorchGradientBoostingClassifier
from .label_binarizer import TorchLabelBinarizer
from .linear_regression import TorchLinearRegression
from .nn import TorchMLPClassifier, TorchMLPRegressor
from .nystroem import TorchNystroem
from .pipeline import TorchPipeline
from .sgd_classifier import TorchSGDClassifier
from .standard_scaler import TorchStandardScaler
from .svc import TorchLinearSVC, TorchSVC
from .svr import TorchLinearSVR, TorchSVR
from .tree import TorchDecisionTreeClassifier, TorchDecisionTreeRegressor
from .ttr import TorchTransformedTargetRegressor

# This list is intentionally kept alphabetical.
_REGISTRY = [
    TorchDecisionTreeClassifier,
    TorchDecisionTreeRegressor,
    TorchDummyClassifier,
    TorchGradientBoostingClassifier,
    TorchLabelBinarizer,
    TorchLinearRegression,
    TorchLinearSVC,
    TorchLinearSVR,
    TorchMLPClassifier,
    TorchMLPRegressor,
    TorchNystroem,
    TorchPipeline,
    TorchSGDClassifier,
    TorchStandardScaler,
    TorchSVC,
    TorchSVR,
    TorchTransformedTargetRegressor,
]


def wrap(obj: BaseEstimator) -> nn.Module:
    for x in _REGISTRY:
        for cls in x.supported_classes():
            if isinstance(obj, cls):
                return x.wrap(obj)
    raise ValueError(f"unsupported sklearn estimator type: {type(obj)}")


def supported_classes() -> List[Any]:
    return [x for y in _REGISTRY for x in y.supported_classes()]
