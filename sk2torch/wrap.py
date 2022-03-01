import torch.nn as nn
from sklearn.base import BaseEstimator

from .nystroem import TorchNystroem
from .pipeline import TorchPipeline
from .sgd_classifier import TorchSGDClassifier
from .standard_scaler import TorchStandardScaler

_REGISTRY = [TorchNystroem, TorchPipeline, TorchSGDClassifier, TorchStandardScaler]


def wrap(obj: BaseEstimator) -> nn.Module:
    for x in _REGISTRY:
        if x.supports_wrap(obj):
            return x.wrap(obj)
    raise ValueError(f"unsupported sklearn estimator type: {type(obj)}")
