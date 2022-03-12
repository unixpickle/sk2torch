from copy import deepcopy
from typing import List, Type

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


class TorchLogisticRegression(nn.Module):
    def __init__(
        self,
        multi_class: str,
        weights: torch.Tensor,
        biases: torch.Tensor,
        classes: torch.Tensor,
    ):
        super().__init__()
        self.multi_class = multi_class
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)
        self.register_buffer("classes", classes)

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [LogisticRegression]

    @classmethod
    def wrap(cls, obj: LogisticRegression) -> "TorchLogisticRegression":
        est = deepcopy(obj)
        est.densify()
        return cls(
            multi_class=est.multi_class,
            weights=torch.from_numpy(est.coef_),
            biases=torch.from_numpy(est.intercept_),
            classes=torch.from_numpy(est.classes_),
        )

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
        outputs = (x @ self.weights.t()) + self.biases
        if outputs.shape[1] == 1:
            return outputs.view(-1)
        return outputs

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
                logits = torch.stack([-logits / 2, logits / 2], dim=-1)
            return F.log_softmax(logits, dim=-1)
