from copy import deepcopy

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier


class TorchSGDClassifier(nn.Module):
    def __init__(self, weights: torch.Tensor, biases: torch.Tensor, loss: str):
        super().__init__()
        self.loss = loss
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    @classmethod
    def supports_wrap(cls, obj: BaseEstimator) -> bool:
        return isinstance(obj, SGDClassifier)

    @classmethod
    def wrap(cls, obj: SGDClassifier) -> "TorchSGDClassifier":
        est = deepcopy(obj)
        est.densify()
        return cls(
            weights=torch.from_numpy(est.coef_),
            biases=torch.from_numpy(est.intercept_),
            loss=est.loss,
        )

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        """
        Predict class labels for the given feature vectors.
        """
        scores = self.decision_function(x)
        if len(scores.shape) == 1:
            return (scores > 0).long()
        else:
            return scores.argmax(-1)

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
        if self.loss == "log":
            logits = self.decision_function(x)
            if len(logits.shape) == 1:
                return torch.stack(
                    [F.logsigmoid(-logits), F.logsigmoid(logits)], dim=-1
                )
            # This is a one-versus-rest classifier.
            return F.log_softmax(F.logsigmoid(logits), dim=-1)
        else:
            assert False, "probability prediction not supported for loss: " + self.loss
