import torch.nn as nn
from sklearn.base import BaseEstimator

from .linear_sgd import TorchSGDClassifier

_REGISTRY = [TorchSGDClassifier]


def wrap(obj: BaseEstimator) -> nn.Module:
    for x in _REGISTRY:
        if x.supports_wrap(obj):
            return x.wrap(obj)
    raise ValueError(f"unsupported sklearn estimator type: {type(obj)}")
