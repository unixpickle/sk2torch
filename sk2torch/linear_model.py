from abc import abstractmethod
from copy import deepcopy
from typing import List, Type, Union

import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.linear_model._base import LinearModel
from sklearn.svm import LinearSVC, LinearSVR


class TorchLinearModel(nn.Module):
    def __init__(
        self,
        model: LinearModel,
    ):
        super().__init__()
        if hasattr(model, "densify"):
            model = deepcopy(model)
            model.densify()
        else:
            assert isinstance(
                model.coef_, np.ndarray
            ), "sparse linear model is not supported"
        weights = torch.from_numpy(model.coef_)
        biases = torch.from_numpy(np.array(model.intercept_))
        if len(weights.shape) == 1:
            weights = weights[None]
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        outputs = (x @ self.weights.t()) + self.biases
        if outputs.shape[1] == 1:
            return outputs.view(-1)
        return outputs


class TorchLinearRegression(TorchLinearModel):
    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [LinearRegression, Ridge, RidgeCV, SGDRegressor, LinearSVR]

    @classmethod
    def wrap(
        cls, obj: Union[LinearRegression, Ridge, RidgeCV, SGDRegressor, LinearSVR]
    ) -> "TorchLinearRegression":
        return cls(obj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the given feature vectors.

        An alias for self.predict().
        """
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the given feature vectors.
        """
        return self._decision_function(x)

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        outputs = (x @ self.weights.t()) + self.biases
        if outputs.shape[1] == 1:
            return outputs.view(-1)
        return outputs


class TorchLinearClassifier(TorchLinearModel):
    def __init__(self, model: LinearModel):
        super().__init__(model=model)
        self.register_buffer("classes", torch.from_numpy(model.classes_))

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [RidgeClassifier, RidgeClassifierCV, LinearSVC]

    @classmethod
    def wrap(
        cls, obj: Union[RidgeClassifier, RidgeClassifierCV, LinearSVC]
    ) -> "TorchLinearClassifier":
        return cls(model=obj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the given feature vectors.

        An alias for self.predict().
        """
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the given feature vectors.
        """
        scores = self.decision_function(x)
        if len(scores.shape) == 1:
            indices = (scores > 0).long()
        else:
            indices = scores.argmax(-1)
        return self.classes[indices]

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        return self._decision_function(x)


class TorchSGDClassifier(TorchLinearClassifier):
    def __init__(
        self,
        loss: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = loss

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [SGDClassifier]

    @classmethod
    def wrap(cls, obj: SGDClassifier) -> "TorchSGDClassifier":
        return cls(
            loss=obj.loss,
            model=obj,
        )

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_log_proba(x).exp()

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.loss == "log":
            logits = self.decision_function(x)
            if len(logits.shape) == 1:
                return torch.stack(
                    [F.logsigmoid(-logits), F.logsigmoid(logits)], dim=-1
                )
            # This is a one-versus-rest classifier.
            return F.log_softmax(F.logsigmoid(logits), dim=-1)
        else:
            raise RuntimeError(
                "probability prediction not supported for loss: " + self.loss
            )


class TorchLogisticRegression(TorchLinearClassifier):
    def __init__(
        self,
        multi_class: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multi_class = multi_class

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [LogisticRegression, LogisticRegressionCV]

    @classmethod
    def wrap(
        cls, obj: Union[LogisticRegression, LogisticRegressionCV]
    ) -> "TorchLogisticRegression":
        multi_class = obj.multi_class
        if multi_class == "auto" and (
            len(obj.classes_) == 2 or obj.solver == "liblinear"
        ):
            multi_class = "ovr"
        return cls(multi_class=multi_class, model=obj)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_log_proba(x).exp()

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decision_function(x)
        if self.multi_class == "ovr":
            if len(logits.shape) == 1:
                return torch.stack(
                    [F.logsigmoid(-logits), F.logsigmoid(logits)], dim=-1
                )
            return F.log_softmax(F.logsigmoid(logits), dim=-1)
        else:
            if len(logits.shape) == 1:
                logits = torch.stack([-logits, logits], dim=-1)
            return F.log_softmax(logits, dim=-1)
