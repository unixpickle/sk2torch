from typing import Any, List

import torch.nn as nn
from sklearn.base import BaseEstimator

from .dummy import TorchDummyClassifier, TorchDummyRegressor
from .gradient_boosting import (
    TorchGradientBoostingClassifier,
    TorchGradientBoostingRegressor,
)
from .label_binarizer import TorchLabelBinarizer
from .linear_model import (
    TorchLinearClassifier,
    TorchLinearRegression,
    TorchLogisticRegression,
    TorchSGDClassifier,
)
from .min_max_scaler import TorchMinMaxScaler
from .nn import TorchMLPClassifier, TorchMLPRegressor
from .nystroem import TorchNystroem
from .pca import TorchPCA
from .pipeline import TorchPipeline
from .stacking import TorchStackingClassifier, TorchStackingRegressor
from .standard_scaler import TorchStandardScaler
from .svc import TorchSVC
from .svr import TorchSVR
from .tree import TorchDecisionTreeClassifier, TorchDecisionTreeRegressor
from .ttr import TorchTransformedTargetRegressor

# This list is intentionally kept alphabetical.
_REGISTRY = [
    TorchDecisionTreeClassifier,
    TorchDecisionTreeRegressor,
    TorchDummyClassifier,
    TorchDummyRegressor,
    TorchGradientBoostingClassifier,
    TorchGradientBoostingRegressor,
    TorchLabelBinarizer,
    TorchLinearClassifier,
    TorchLinearRegression,
    TorchLogisticRegression,
    TorchMLPClassifier,
    TorchMLPRegressor,
    TorchMinMaxScaler,
    TorchNystroem,
    TorchPCA,
    TorchPipeline,
    TorchSGDClassifier,
    TorchStackingClassifier,
    TorchStackingRegressor,
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
