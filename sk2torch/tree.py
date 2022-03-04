from typing import Any, List, Optional, Type

import numpy as np
import torch
import torch.jit
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor


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
        self.value = nn.Parameter(torch.from_numpy(tree.value[is_leaf]).flatten(1))
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

    def _raw_values(self, x: torch.Tensor) -> torch.Tensor:
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
        y = self._raw_values(x)
        if y.shape[1] == 1:
            y = y.view(-1)
        return y
