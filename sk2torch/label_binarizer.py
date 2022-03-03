from typing import List, Optional, Type

import torch
import torch.jit
import torch.nn as nn
from sklearn.preprocessing import LabelBinarizer


class TorchLabelBinarizer(nn.Module):
    def __init__(
        self,
        classes: torch.Tensor,
        neg_label: int,
        pos_label: int,
        y_type: str,
    ):
        super().__init__()
        self.register_buffer("classes", classes)
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.y_type = y_type

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [LabelBinarizer]

    @classmethod
    def wrap(cls, obj: LabelBinarizer) -> "TorchLabelBinarizer":
        assert obj.y_type_ in ["multiclass", "multilabel-indicator", "binary"]
        return cls(
            classes=torch.from_numpy(obj.classes_),
            pos_label=int(obj.pos_label),
            neg_label=int(obj.neg_label),
            y_type=obj.y_type_,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.y_type == "multilabel-indicator":
            # The logic in the sklearn code is pretty inconsistent and broken
            # when using arbitrary pos_label/neg_label, but we replicate its
            # behavior for this case as closely as possible.
            # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/preprocessing/_label.py#L549-L566
            pos_label = self.pos_label
            pos_switch = pos_label == 0
            if pos_switch:
                pos_label = -self.neg_label
            if pos_label != 1:
                # We preserve x's dtype here, which means pos_label will be
                # cast to it. For example, if x is a bool, and pos_label > 1,
                # then pos_label will be forced to 1.
                # This is a bug in the scikit-learn implementation, so we
                # immitate it.
                x = torch.where(x != 0, torch.tensor(pos_label).to(x), x)
            x = x.long()
            if self.neg_label != 0:
                x = torch.where(x == 0, self.neg_label, x)
            if pos_switch:
                x = torch.where(x == pos_label, 0, x)
            return x
        elif self.y_type == "multiclass":
            return torch.where(
                x[..., None] == self.classes, self.pos_label, self.neg_label
            ).long()
        else:
            assert self.y_type == "binary"
            return torch.where(
                x[..., None] == self.classes[1:], self.pos_label, self.neg_label
            ).long()

    @torch.jit.export
    def inverse_transform(
        self, x: torch.Tensor, threshold: Optional[float] = None
    ) -> torch.Tensor:
        if self.y_type == "multiclass":
            return self.classes[x.argmax(-1)]
        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2
        outputs = self.classes[(x > threshold).long()]
        if self.y_type == "binary":
            outputs = outputs.view(-1)
        return outputs
