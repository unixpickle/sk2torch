from typing import List, Type, Union

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .label_binarizer import TorchLabelBinarizer


class _WrappedMLP(nn.Module):
    def __init__(self, layers: nn.Sequential, out_act: str):
        super().__init__()
        assert out_act in ["identity", "softmax", "logistic"]
        self.layers = layers
        self.out_act = out_act

    @classmethod
    def wrap(cls, obj: Union[MLPClassifier, MLPRegressor]) -> "_WrappedMLP":
        acts = {
            "identity": nn.Identity,
            "logistic": nn.Sigmoid,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
        }
        if obj.activation not in acts:
            raise ValueError(f"unsupported activation: {obj.activation}")
        act = acts[obj.activation]
        modules = []
        for weights, biases in zip(obj.coefs_, obj.intercepts_):
            w = torch.from_numpy(weights)
            b = torch.from_numpy(biases)
            module = nn.Linear(weights.shape[0], weights.shape[1], dtype=w.dtype)
            with torch.no_grad():
                module.weight.copy_(w.t())
                module.bias.copy_(b)
            modules.append(module)
            modules.append(act())
        del modules[-1]  # output activation handled separately.
        return cls(layers=nn.Sequential(*modules), out_act=obj.out_activation_)

    def forward(self, x: torch.Tensor, include_negative: bool = False) -> torch.Tensor:
        x = self.layers(x)
        if x.shape[1] == 1 and self.out_act == "logistic" and include_negative:
            x = torch.cat([F.logsigmoid(-x), F.logsigmoid(x)], dim=-1)
        elif self.out_act == "logistic":
            x = F.logsigmoid(x)
        elif self.out_act == "softmax":
            x = F.log_softmax(x, dim=-1)
        return x


class TorchMLPClassifier(nn.Module):
    def __init__(
        self,
        module: _WrappedMLP,
        label_binarizer: TorchLabelBinarizer,
    ):
        super().__init__()
        self.module = module
        self.label_binarizer = label_binarizer

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [MLPClassifier]

    @classmethod
    def wrap(cls, obj: MLPClassifier) -> "TorchMLPClassifier":
        return cls(
            module=_WrappedMLP.wrap(obj),
            label_binarizer=TorchLabelBinarizer.wrap(obj._label_binarizer),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.module(x).exp()
        return self.label_binarizer.inverse_transform(probs)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_log_proba(x).exp()

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x, include_negative=True)


class TorchMLPRegressor(nn.Module):
    def __init__(
        self,
        module: _WrappedMLP,
    ):
        super().__init__()
        self.module = module

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [MLPRegressor]

    @classmethod
    def wrap(cls, obj: MLPRegressor) -> "TorchMLPRegressor":
        return cls(module=_WrappedMLP.wrap(obj))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        if out.shape[1] == 1:
            return out.view(-1)
        return out
