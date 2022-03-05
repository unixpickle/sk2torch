from typing import Any, List, Optional, Type, Union

import numpy as np
import torch
import torch.jit
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaseTree(nn.Module):
    def __init__(
        self,
        tree: Any,
    ):
        super().__init__()

        # Track which nodes are branches/leaves. This will be used to index
        # matrices to avoid storing unused values.
        is_branch = np.zeros(len(tree.value), dtype=bool)

        # Rows are leaves and columns are branches in the next two arrays.
        # The conditions bitmap indicates, for a given leaf, what branches
        # must be true.
        # The mask bitmap indicates, for a given leaf, what branches must be
        # checked (i.e. ignore branches outside of a leaf's path).
        node_conditions = np.zeros([len(tree.value)] * 2, dtype=bool)
        node_conditions_mask = node_conditions.copy()

        # Cache of the used nodes for each branch.
        # This is just node_conditions_mask with the diagonal set to True.
        decision_paths = node_conditions.copy()

        def enumerate_tree(node_id: int, parent_id: Optional[int] = None):
            if parent_id is not None:
                node_conditions_mask[node_id] = node_conditions_mask[parent_id]
                node_conditions_mask[node_id][parent_id] = True
                decision_paths[node_id] = decision_paths[parent_id]
            decision_paths[node_id, node_id] = True
            left_id, right_id = (
                tree.children_left[node_id],
                tree.children_right[node_id],
            )
            if left_id != right_id:
                is_branch[node_id] = True
                node_conditions[left_id] = node_conditions[node_id]
                node_conditions[right_id] = node_conditions[node_id]
                node_conditions[right_id][node_id] = True
                enumerate_tree(left_id, node_id)
                enumerate_tree(right_id, node_id)

        enumerate_tree(0)

        # We will perform accumulations (sometimes via matmul) along each row
        # of these matrices, so we must be able to store the largest sum.
        max_depth = np.max(np.sum(node_conditions_mask.astype(np.int64), axis=-1))
        if max_depth < 2 ** 7:
            mat_dtype = torch.int8
        elif max_depth < 2 ** 15:
            mat_dtype = torch.int16
        else:
            mat_dtype = torch.int32

        is_leaf = np.logical_not(is_branch)
        self.register_buffer("feature", torch.from_numpy(tree.feature[is_branch]))
        self.value = nn.Parameter(torch.from_numpy(tree.value[is_leaf]))
        self.threshold = nn.Parameter(torch.from_numpy(tree.threshold[is_branch]))
        self.register_buffer(
            "cond",
            torch.from_numpy(node_conditions[np.ix_(is_leaf, is_branch)]).to(mat_dtype),
        )
        self.register_buffer(
            "cond_mask",
            torch.from_numpy(node_conditions_mask[np.ix_(is_leaf, is_branch)]).to(
                mat_dtype
            ),
        )
        self.register_buffer(
            "decision_paths", torch.from_numpy(decision_paths[is_leaf])
        )

    def _leaf_indices(self, x: torch.Tensor) -> torch.Tensor:
        comparisons = (x[:, self.feature] > self.threshold).to(self.cond)
        cond_counts = comparisons @ self.cond.t()
        # Two conditions:
        # 1. no_false_neg: every condition that needs to be true is true.
        # 2. no_false_pos: no other conditions along the decision path are true.
        no_false_neg = cond_counts == self.cond.sum(1)
        no_false_pos = cond_counts == (comparisons @ self.cond_mask.t())
        return (no_false_neg & no_false_pos).int().argmax(-1)

    @torch.jit.export
    def raw_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.value[self._leaf_indices(x)]

    @torch.jit.export
    def decision_path(self, x: torch.Tensor):
        return self.decision_paths[self._leaf_indices(x)]


class TorchDecisionTreeRegressor(BaseTree):
    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [DecisionTreeRegressor]

    @classmethod
    def wrap(cls, obj: DecisionTreeRegressor) -> "TorchDecisionTreeRegressor":
        return TorchDecisionTreeRegressor(obj.tree_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self.raw_values(x).view(len(x), -1)
        if y.shape[1] == 1:
            y = y.view(-1)
        return y


class TorchDecisionTreeClassifier(BaseTree):
    def __init__(
        self,
        n_outputs: int,
        n_classes: Union[int, List[int]],
        classes: Union[List[torch.Tensor], torch.Tensor],
        tree: Any,
    ):
        super().__init__(tree)
        self.n_outputs = int(n_outputs)
        if n_outputs == 1:
            self.outputs = nn.ModuleList([_SingleClassOutput(n_classes, classes)])
        else:
            self.outputs = nn.ModuleList(
                [_SingleClassOutput(x, y) for x, y in zip(n_classes, classes)]
            )

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [DecisionTreeClassifier]

    @classmethod
    def wrap(cls, obj: DecisionTreeClassifier) -> "TorchDecisionTreeClassifier":
        if obj.n_outputs_ == 1:
            classes = torch.from_numpy(obj.classes_)
        else:
            classes = [torch.from_numpy(x) for x in obj.classes_]
        return TorchDecisionTreeClassifier(
            obj.n_outputs_, obj.n_classes_, classes, obj.tree_
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y = self.raw_values(x)
        outputs = []
        for i, x in enumerate(self.outputs):
            outputs.append(x(y[:, i]))
        if self.n_outputs == 1:
            return outputs[0]
        else:
            return torch.stack(outputs, dim=-1)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        y = self.raw_values(x)
        outputs = []
        for i, x in enumerate(self.outputs):
            outputs.append(x.predict_proba(y[:, i]))
        return self._collapse(outputs)

    @torch.jit.export
    def predict_log_proba(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        y = self.raw_values(x)
        outputs = []
        for i, x in enumerate(self.outputs):
            outputs.append(x.predict_log_proba(y[:, i]))
        return self._collapse(outputs)

    def _collapse(
        self, x: List[torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.n_outputs == 1:
            return x[0]
        else:
            return x


class _SingleClassOutput(nn.Module):
    def __init__(self, n_classes: int, classes: torch.Tensor):
        super().__init__()
        self.n_classes = int(n_classes)
        self.register_buffer("classes", classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classes[x[:, : self.n_classes].argmax(1)]

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        proba = x[:, : self.n_classes]
        normalizer = proba.sum(1, keepdim=True)
        proba = proba / torch.where(normalizer == 0.0, 1.0, normalizer)
        return proba

    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).log()
