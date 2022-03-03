import numpy as np
import pytest
import torch
import torch.jit
from sklearn.preprocessing import LabelBinarizer

from .label_binarizer import TorchLabelBinarizer


@pytest.mark.parametrize(
    ("inputs", "neg_label", "pos_label"),
    [
        (np.array([1, 0, 1, 1, 0]), 0, 1),
        (np.array([True, False, True, True, False]), 0, 1),
        (np.array([2, 1, 2, 2, 1]), 0, 1),
        (np.array([1, 0, 1, 1, 0]), 1, 2),
        (np.array([2, 1, 2, 2, 1]), 1, 2),
        (np.array([True, False, True, True, False]), 1, 2),
        (np.array([1, 0, 1, 1, 0]), -1, 0),
        (np.array([1, 0, 1, 1, 0]), -13, 13),
        (np.array([[1, 0], [1, 1], [0, 0]]), 0, 1),
        (np.array([[1, 2], [2, 2], [1, 1]]), 0, 1),
        (np.array([[1, 5], [5, 5], [1, 1]]), 0, 1),
        (np.array([[1, 0], [1, 1], [0, 0]]), 1, 2),
        (np.array([[1, 2], [2, 1], [1, 2]]), 1, 2),
        (np.array([[-1, 5], [5, -1], [-1, 5]]), 1, 2),
        (np.array([[1, 0], [1, 1], [0, 0]]), -1, 0),
        (np.array([[True, False], [True, True], [False, False]]), 0, 1),
        (np.array([[True, False], [True, True], [False, False]]), 1, 2),
        (np.array([[True, False], [True, True], [False, False]]), 3, 4),
        (np.array([[True, False], [True, True], [False, False]]), -1, 0),
        (np.array([0, 1, 2, 3]), 0, 1),
        (np.array([6, 10, 12, 24]), 0, 1),
        (np.array([0, 1, 2, 3]), 1, 2),
        (np.array([6, 10, 12, 24]), 1, 2),
        (np.array([0, 1, 2, 3]), -1, 0),
    ],
)
def test_label_binarizer(inputs: np.ndarray, neg_label: int, pos_label: int):
    sk_obj = LabelBinarizer(neg_label=neg_label, pos_label=pos_label)
    sk_obj.fit(inputs)
    th_obj = torch.jit.script(TorchLabelBinarizer.wrap(sk_obj))

    inputs_th = torch.from_numpy(inputs)

    with torch.no_grad():
        expected = sk_obj.transform(inputs)
        actual = th_obj.transform(inputs_th).numpy()
        assert (expected == actual).all()

        expected = sk_obj.inverse_transform(sk_obj.transform(inputs))
        actual = th_obj.inverse_transform(th_obj.transform(inputs_th)).numpy()
        assert (expected == actual).all()

        xf_shape = sk_obj.transform(inputs).shape
        random_th = torch.rand(100, *xf_shape[1:]).double()
        expected = sk_obj.inverse_transform(random_th.numpy())
        actual = th_obj.inverse_transform(random_th).numpy()
        assert (expected == actual).all()
