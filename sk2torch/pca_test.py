# type: ignore

import numpy as np
import pytest
import torch
import torch.jit
from sklearn.decomposition import PCA

from .pca import TorchPCA


@pytest.mark.parametrize(("whiten",), [(False,), (True,)])
def test_pca(whiten: bool):
    rng = np.random.RandomState(1337)
    xs = (rng.normal(size=(1000, 4)) @ rng.normal(size=(4, 10))) + rng.normal(
        size=(10,)
    )
    sk_obj = PCA(n_components=3, whiten=whiten)
    sk_obj.fit(xs)
    th_obj = torch.jit.script(TorchPCA.wrap(sk_obj))

    xs_th = torch.from_numpy(xs)

    with torch.no_grad():
        actual = th_obj.transform(xs_th).numpy()
        expected = sk_obj.transform(xs)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

        ys = expected
        actual = th_obj.inverse_transform(torch.from_numpy(ys)).numpy()
        expected = sk_obj.inverse_transform(ys)
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)
