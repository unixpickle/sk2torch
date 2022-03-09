from typing import List, Type, Union

import numpy as np
import torch
import torch.jit
import torch.nn as nn
from sklearn.dummy import DummyClassifier, DummyRegressor


class TorchDummyClassifier(nn.Module):
    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [DummyClassifier]

    @classmethod
    def wrap(
        cls, obj: DummyClassifier
    ) -> Union["TorchDummyClassifierSingle", "TorchDummyClassifierMulti"]:
        assert not obj.sparse_output_, "sparse classifiers are not supported"
        if isinstance(obj.n_classes_, list):
            return TorchDummyClassifierMulti(
                singles=[
                    TorchDummyClassifierSingle(
                        classes=torch.from_numpy(obj.classes_[i]),
                        class_prior=torch.from_numpy(obj.class_prior_[i]),
                        strategy=obj.strategy,
                        constant=(
                            torch.from_numpy(np.array(obj.constant[i]))
                            if obj.constant is not None
                            else torch.from_numpy(obj.classes_[i])[0]
                        ),
                    )
                    for i in range(len(obj.n_classes_))
                ],
            )
        else:
            return TorchDummyClassifierSingle(
                classes=torch.from_numpy(obj.classes_),
                class_prior=torch.from_numpy(obj.class_prior_),
                strategy=obj.strategy,
                constant=(
                    torch.from_numpy(np.array(obj.constant))
                    if obj.constant is not None
                    else torch.from_numpy(obj.classes_)[0]
                ),
            )


class TorchDummyClassifierSingle(TorchDummyClassifier):
    def __init__(
        self,
        classes: torch.Tensor,
        class_prior: torch.Tensor,
        strategy: str,
        constant: torch.Tensor,
    ):
        super().__init__()
        assert strategy in [
            "most_frequent",
            "prior",
            "stratified",
            "uniform",
            "constant",
        ]
        self.register_buffer("classes", classes)
        self.class_prior = nn.Parameter(class_prior)
        self.strategy = strategy
        self.register_buffer("constant", constant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.strategy == "most_frequent" or self.strategy == "prior":
            return self.classes[self.class_prior.argmax()].repeat(len(x))
        elif self.strategy == "stratified":
            return self.classes[
                torch.multinomial(
                    self.class_prior, num_samples=len(x), replacement=True
                )
            ]
        elif self.strategy == "uniform":
            return self.classes[
                torch.randint(low=0, high=len(self.classes), size=(len(x),))
            ]
        return self.constant.view(1).repeat(len(x))

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.strategy == "most_frequent":
            out = self.classes == self.classes[self.class_prior.argmax()]
            return out[None].repeat(len(x), 1).to(self.class_prior)
        elif self.strategy == "prior":
            return self.class_prior[None].repeat(len(x), 1)
        elif self.strategy == "stratified":
            samples = torch.multinomial(
                self.class_prior, num_samples=len(x), replacement=True
            )
            return (self.classes == self.classes[samples[:, None]]).to(self.class_prior)
        elif self.strategy == "uniform":
            out = torch.ones_like(self.class_prior) / len(self.class_prior)
            return out[None].repeat(len(x), 1)
        out = self.classes == self.constant
        return out[None].repeat(len(x), 1).to(self.class_prior)

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).log()


class TorchDummyClassifierMulti(TorchDummyClassifier):
    def __init__(self, singles: List[TorchDummyClassifierSingle]):
        super().__init__()
        self.singles = nn.ModuleList(singles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([single.predict(x) for single in self.singles], dim=1)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [single.predict_proba(x) for single in self.singles]

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        return [single.predict_log_proba(x) for single in self.singles]


class TorchDummyRegressor(nn.Module):
    def __init__(self, strategy: str, constant: torch.Tensor):
        super().__init__()
        if strategy == "constant":
            self.register_buffer("constant", constant)
        else:
            self.constant = nn.Parameter(constant)

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [DummyRegressor]

    @classmethod
    def wrap(cls, obj: DummyRegressor) -> "TorchDummyRegressor":
        return cls(
            strategy=obj.strategy, constant=torch.from_numpy(obj.constant_).view(-1)
        )

    def forward(self, x: torch.Tensor):
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        res = self.constant[None].repeat(len(x), 1)
        if res.shape[1] == 1:
            res = res.view(-1)
        return res
