from typing import List, Optional, Type

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier


class _GradientBoostingStage(nn.Module):
    def __init__(self, trees: List[BaseEstimator]):
        super().__init__()
        from .wrap import wrap

        self.trees = nn.ModuleList([wrap(x) for x in trees])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([tree(x).view(-1) for tree in self.trees], dim=-1)


class TorchGradientBoostingClassifier(nn.Module):
    def __init__(
        self,
        stages: List[_GradientBoostingStage],
        classes: torch.Tensor,
        init: Optional[nn.Module],
        loss: str,
        learning_rate: float,
    ):
        super().__init__()
        assert loss in ["exponential", "deviance"]
        self.stages = nn.ModuleList(stages)
        self.register_buffer("classes", classes)
        self.has_init = init is not None
        if self.has_init:
            self.init = init
        else:
            # Needed to prevent TorchScript from erroring.
            self.init = nn.Identity()

            def dummy_fn(x: torch.Tensor) -> torch.Tensor:
                return x

            self.init.predict_proba = dummy_fn
        self.loss = loss
        self.learning_rate = learning_rate

        dimension = len(stages[0].trees)
        param = next(self.parameters())
        self.register_buffer(
            "zero_out",
            torch.zeros(dimension, dtype=param.dtype, device=param.device),
        )

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [GradientBoostingClassifier]

    @classmethod
    def wrap(cls, obj: GradientBoostingClassifier) -> "TorchGradientBoostingClassifier":
        from .wrap import wrap

        return cls(
            stages=[_GradientBoostingStage(x) for x in obj.estimators_.tolist()],
            classes=torch.from_numpy(obj.classes_),
            init=wrap(obj.init_) if obj.init_ != "zero" else None,
            loss=obj.loss,
            learning_rate=obj.learning_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.loss == "exponential":
            decisions = self.decision_function(x)
            assert len(decisions.shape) == 1
            return self.classes[(decisions >= 0).long()]
        proba = self.predict_proba(x)
        return self.classes[proba.argmax(-1)]

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_log_proba(x).exp()

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        decisions = self.decision_function(x)
        if self.loss == "deviance":
            if len(decisions.shape) == 1:
                return torch.stack(
                    [F.logsigmoid(-decisions), F.logsigmoid(decisions)], dim=-1
                )
            else:
                return F.log_softmax(decisions, dim=-1)
        else:
            return torch.stack(
                [F.logsigmoid(-2 * decisions), F.logsigmoid(2 * decisions)], dim=-1
            )

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        out = self._init_outputs(x)
        for stage in self.stages:
            out = out + stage.forward(x) * self.learning_rate
        if out.shape[1] == 1:
            return out.view(-1)
        return out

    def _init_outputs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_init:
            return self.zero_out[None].repeat(len(x), 1)
        eps = 1.1920929e-07  # np.finfo(np.float32).eps
        init_probs = self.init.predict_proba(x).clamp(eps, 1 - eps)
        if self.loss == "exponential":
            assert init_probs.shape[1] == 2
            # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/ensemble/_gb_losses.py#L978
            prob_pos = init_probs[:, 1]
            return (0.5 * (prob_pos / (1 - prob_pos)).log())[:, None]
        else:
            if init_probs.shape[1] == 2:
                # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/ensemble/_gb_losses.py#L749
                prob_pos = init_probs[:, 1]
                return ((prob_pos / (1 - prob_pos)).log())[:, None]
            else:
                # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/ensemble/_gb_losses.py#L866
                return init_probs.log()
