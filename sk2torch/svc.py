from typing import List, Optional, Tuple, Type, Union

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC, NuSVC

from .kernel import Kernel


class TorchSVC(nn.Module):
    def __init__(
        self,
        kernel: Kernel,
        ovr: bool,
        break_ties: bool,
        n_support: List[int],
        support_vectors: torch.Tensor,
        intercept: torch.Tensor,
        dual_coef: torch.Tensor,
        classes: torch.Tensor,
        prob_a: Optional[torch.Tensor],
        prob_b: Optional[torch.Tensor],
    ):
        super().__init__()
        self.kernel = kernel
        self.ovr = ovr
        self.break_ties = break_ties
        self.n_support = n_support
        self.support_vectors = nn.Parameter(support_vectors)
        self.intercept = nn.Parameter(intercept)
        self.dual_coef = nn.Parameter(dual_coef)
        self.register_buffer("classes", classes)
        self.supports_prob = prob_a is not None and prob_b is not None
        self.prob_a = nn.Parameter(
            torch.ones_like(intercept) if prob_a is None else prob_a
        )
        self.prob_b = nn.Parameter(
            torch.zeros_like(intercept) if prob_b is None else prob_b
        )
        self.n_classes = len(n_support)

        self.sv_offsets = []
        offset = 0
        for count in n_support:
            self.sv_offsets.append(offset)
            offset += count

        self._ovo_index_map = [-1 for _ in range(self.n_classes ** 2)]
        k = 0
        for i in range(self.n_classes - 1):
            for j in range(i + 1, self.n_classes):
                self._ovo_index_map[i * self.n_classes + j] = k
                self._ovo_index_map[j * self.n_classes + i] = k
                k += 1

    @classmethod
    def supported_classes(cls) -> List[Type]:
        return [SVC, NuSVC]

    @classmethod
    def wrap(cls, obj: Union[SVC, NuSVC]) -> "TorchSVC":
        assert not obj._sparse, "sparse SVC not supported"
        assert obj.decision_function_shape in ["ovo", "ovr"]
        return cls(
            kernel=Kernel.wrap(obj),
            ovr=obj.decision_function_shape == "ovr",
            break_ties=obj.break_ties,
            n_support=obj.n_support_.tolist(),
            support_vectors=torch.from_numpy(obj.support_vectors_),
            intercept=torch.from_numpy(obj.intercept_),
            dual_coef=torch.from_numpy(obj.dual_coef_),
            classes=torch.from_numpy(obj.classes_),
            prob_a=torch.from_numpy(obj.probA_) if obj.probability else None,
            prob_b=torch.from_numpy(obj.probB_) if obj.probability else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ovo, ovr = self.decision_function_ovo_ovr(x)
        if self.n_classes == 2:
            indices = (ovo.view(-1) > 0).long()
        elif self.ovr and self.break_ties:
            indices = ovr.argmax(dim=-1)
        else:
            indices = ovr.round().argmax(dim=-1)
        return self.classes[indices]

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        ovo, ovr = self.decision_function_ovo_ovr(x)
        if len(self.n_support) == 2:
            return ovo.view(-1)
        elif self.ovr:
            return ovr
        else:
            return ovo

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        assert self.supports_prob, "model must be trained with probability=True"
        if self.n_classes == 2:
            ovo, _ = self.decision_function_ovo_ovr(x)
            logit = ovo * self.prob_a - self.prob_b
            return torch.cat([F.logsigmoid(logit), F.logsigmoid(-logit)], dim=-1)
        return self.predict_proba(x).log()

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        assert self.supports_prob, "model must be trained with probability=True"
        ovo, _ = self.decision_function_ovo_ovr(x)
        if self.n_classes == 2:
            # This shortcut optimization is present in the latest LibSVM
            # but not in the scikit-learn fork. As a result, the scikit
            # version is slightly less accurate in the binary case.
            # https://github.com/scikit-learn/scikit-learn/blob/82df48934eba1df9a1ed3be98aaace8eada59e6e/sklearn/svm/src/libsvm/svm.cpp#L2925

            # For some reason, prob_a has the opposite sign for this case.
            logit = ovo * self.prob_a - self.prob_b
            return torch.cat([logit.sigmoid(), (-logit).sigmoid()], dim=-1)

        probs = (-ovo * self.prob_a - self.prob_b).sigmoid()
        min_prob = 1e-7
        probs = probs.clamp(min_prob, 1 - min_prob)

        matrix = self._prob_matrix(probs)
        inv_diag = 1 / (torch.diagonal(matrix, dim1=1, dim2=2) + 1e-12)
        guess = torch.ones(len(x), self.n_classes).to(matrix) / self.n_classes
        masks = torch.eye(self.n_classes).to(matrix)

        # We must use variable name `i` and pre-define `delta`
        # to appease the TorchScript compiler. For the loop, the
        # variable name "_" does not work.
        delta = torch.zeros_like(guess)
        for i in range(max(100, self.n_classes)):
            # Coordinate descent is performed for each dimension between
            # checks of the stopping criterion:
            # https://github.com/cjlin1/libsvm/blob/eedbb147ea79af44f2ecdca1064f2c6a8fe6462d/svm.cpp#L1873-L1883
            for coord in range(self.n_classes):
                mask = masks[coord]
                # Eq 47 from https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf
                mg = (matrix @ guess[:, :, None]).view(guess.shape)
                outer = torch.einsum("ij,ij->i", guess, mg)[:, None]
                delta = outer - mg

                guess = guess + inv_diag * delta * mask
                guess = guess / guess.sum(dim=-1, keepdim=True)

            # Stopping criterion.
            if delta.abs().max().item() < 0.005 / self.n_classes:
                break
        return guess

    def _prob_matrix(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise probability matrix Q from page 31 of
        https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf.
        """
        matrix_elems = []
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i == j:
                    prob_sum = torch.zeros_like(probs[:, 0])
                    for k in range(self.n_classes):
                        if k != i:
                            prob_sum = prob_sum + self._pairwise_prob(probs, k, i) ** 2
                    matrix_elems.append(prob_sum)
                else:
                    matrix_elems.append(
                        -self._pairwise_prob(probs, i, j)
                        * self._pairwise_prob(probs, j, i)
                    )
        return torch.stack(matrix_elems, dim=-1).view(
            len(probs), self.n_classes, self.n_classes
        )

    def _pairwise_prob(self, probs: torch.Tensor, i: int, j: int) -> torch.Tensor:
        p = probs[:, self._ovo_index_map[i * self.n_classes + j]]
        if i > j:
            return 1 - p
        return p

    def decision_function_ovo_ovr(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the one-versus-one and one-versus-rest decision functions."""
        kernel_out = self.kernel(x, self.support_vectors)

        votes = torch.zeros((len(x), self.n_classes)).to(kernel_out)
        confidence_sum = [
            torch.zeros(len(x)).to(kernel_out) for i in range(self.n_classes)
        ]
        ovos = []
        k = 0
        for i in range(self.n_classes - 1):
            for j in range(i + 1, self.n_classes):
                neg_confidence = (
                    self._dual_sum(kernel_out, i, j)
                    + self._dual_sum(kernel_out, j, i)
                    + self.intercept[k]
                )
                k += 1
                ovos.append(neg_confidence)
                confidence_sum[i] = confidence_sum[i] + neg_confidence
                confidence_sum[j] = confidence_sum[j] - neg_confidence
                pred = neg_confidence < 0
                votes[:, i] += torch.logical_not(pred)
                votes[:, j] += pred
        ovo = torch.stack(ovos, dim=-1)
        confidences = torch.stack(confidence_sum, dim=-1)

        # Don't overrule votes while allowing confidences to break ties.
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/utils/multiclass.py#L475
        confidences = confidences / (3 * (confidences.abs() + 1))

        return ovo, votes + confidences

    def _dual_sum(self, kernel_out: torch.Tensor, i: int, j: int) -> torch.Tensor:
        assert j != i
        if j > i:
            j -= 1
        offset, count = self.sv_offsets[i], self.n_support[i]
        coeffs = self.dual_coef[j, offset : offset + count]
        kernel_row = kernel_out[:, offset : offset + count]
        return kernel_row @ coeffs
