import torch.nn as nn
from sklearn.base import BaseEstimator

from .label_binarizer import TorchLabelBinarizer
from .nystroem import TorchNystroem
from .pipeline import TorchPipeline
from .sgd_classifier import TorchSGDClassifier
from .standard_scaler import TorchStandardScaler
from .svc import TorchLinearSVC, TorchSVC
from .svr import TorchLinearSVR, TorchSVR

_REGISTRY = [
    TorchLabelBinarizer,
    TorchLinearSVC,
    TorchLinearSVR,
    TorchNystroem,
    TorchPipeline,
    TorchSGDClassifier,
    TorchStandardScaler,
    TorchSVC,
    TorchSVR,
]


def wrap(obj: BaseEstimator) -> nn.Module:
    for x in _REGISTRY:
        if x.supports_wrap(obj):
            return x.wrap(obj)
    raise ValueError(f"unsupported sklearn estimator type: {type(obj)}")
