from typing import List, Type

import torch
import torch.jit
import torch.nn as nn
from sklearn.ensemble import StackingClassifier, StackingRegressor

from .util import fill_unsupported


class TorchStackingClassifier(nn.Module):
    def __init__(
        self,
        passthrough: bool,
        estimators: List[nn.Module],
        stack_methods: List[str],
        final_estimator: nn.Module,
        classes: torch.Tensor,
    ):
        super().__init__()
        self.passthrough = passthrough
        self.estimators = nn.ModuleList(estimators)
        for model in self.estimators:
            fill_unsupported(model, "predict_proba", "decision_function", "predict")
        self.stack_methods = stack_methods
        self.final_estimator = final_estimator
        fill_unsupported(
            self.final_estimator, "predict_proba", "decision_function", "predict"
        )
        self.register_buffer("classes", classes)

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [StackingClassifier]

    @classmethod
    def wrap(cls, obj: StackingClassifier) -> "TorchStackingClassifier":
        from .wrap import wrap

        return cls(
            passthrough=obj.passthrough,
            estimators=[wrap(x) for x in obj.estimators_],
            stack_methods=obj.stack_method_,
            final_estimator=wrap(obj.final_estimator_),
            classes=torch.from_numpy(obj.classes_),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.classes[self.final_estimator.predict(self.transform(x))]

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_estimator.predict_proba(self.transform(x))

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_estimator.decision_function(self.transform(x))

    @torch.jit.export
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, estimator in enumerate(self.estimators):
            method = self.stack_methods[i]
            if method == "predict":
                out = estimator.predict(x)
            elif method == "predict_proba":
                out = estimator.predict_proba(x)
                if out.shape[1] == 2:
                    out = out[:, 1:]
            else:
                assert method == "decision_function"
                out = estimator.decision_function(x)
            if len(out.shape) == 1:
                out = out[:, None]
            outputs.append(out)
        if self.passthrough:
            outputs.append(x.view(len(x), -1))
        return torch.cat(outputs, dim=-1).to(x)


class TorchStackingRegressor(nn.Module):
    def __init__(
        self,
        passthrough: bool,
        estimators: List[nn.Module],
        final_estimator: nn.Module,
    ):
        super().__init__()
        self.passthrough = passthrough
        self.estimators = nn.ModuleList(estimators)
        self.final_estimator = final_estimator

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [StackingRegressor]

    @classmethod
    def wrap(cls, obj: StackingRegressor) -> "TorchStackingRegressor":
        from .wrap import wrap

        return cls(
            passthrough=obj.passthrough,
            estimators=[wrap(x) for x in obj.estimators_],
            final_estimator=wrap(obj.final_estimator_),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_estimator.predict(self.transform(x))

    @torch.jit.export
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, estimator in enumerate(self.estimators):
            outputs.append(estimator.predict(x).view(-1, 1))
        if self.passthrough:
            outputs.append(x.view(len(x), -1))
        return torch.cat(outputs, dim=-1).to(x)
