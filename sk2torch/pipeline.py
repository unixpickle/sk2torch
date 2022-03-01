from typing import List, Tuple

import torch
import torch.jit
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class TorchPipeline(nn.Module):
    def __init__(self, stages: List[Tuple[str, nn.Module]]):
        super().__init__()
        self.transforms = nn.ModuleDict({k: v for k, v in stages[:-1]})
        for transform in self.transforms.values():
            _fill_unsupported(transform, "inverse_transform")

        self.final_name, self.final = stages[-1]
        _fill_unsupported(
            self.final,
            "decision_function",
            "predict",
            "predict_proba",
            "predict_log_proba",
            "transform",
            "inverse_transform",
        )

    @classmethod
    def supports_wrap(cls, obj: BaseEstimator) -> bool:
        return isinstance(obj, Pipeline)

    @classmethod
    def wrap(cls, obj: Pipeline) -> "TorchPipeline":
        from .wrap import wrap

        mapping = []
        for k, v in obj.steps:
            mapping.append((k, wrap(v)))
        return cls(mapping)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the pipeline and call forward() on the final model.
        """
        return self.final(self._run_transforms(x))

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the pipeline and call predict() on the final model.
        """
        return self.final.predict(self._run_transforms(x))

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the pipeline and call decision_function() on the final model.
        """
        return self.final.decision_function(self._run_transforms(x))

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the pipeline and call predict_proba() on the final model.
        """
        return self.final.predict_proba(self._run_transforms(x))

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the pipeline and call predict_log_proba() on the final model.
        """
        return self.final.predict_log_proba(self._run_transforms(x))

    @torch.jit.export
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the pipeline and call transform() on the final model.
        """
        return self.final.transform(self._run_transforms(x))

    def _run_transforms(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms.values():
            x = transform(x)
        return x

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the pipeline and call inverse_transform() on the final model.
        """
        x = self.final.inverse_transform(x)
        for transform in self.transforms.values()[::-1]:
            x = transform.inverse_transform(x)
        return x


def _fill_unsupported(module: nn.Module, *names: str):
    for name in names:
        if not hasattr(module, name):

            def unsupported_fn(
                _: torch.Tensor, unsup_method_name: str = name
            ) -> torch.Tensor:
                raise RuntimeError(
                    f"method {unsup_method_name} is not supported on this object"
                )

            setattr(module, name, unsupported_fn)
