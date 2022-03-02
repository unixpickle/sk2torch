# sk2torch

**sk2torch** converts [scikit-learn](https://scikit-learn.org/) models into [PyTorch](https://pytorch.org/) modules that can be tuned with backpropagation and even compiled as [TorchScript](https://pytorch.org/docs/stable/jit.html).

Problems solved by this project:
 1. scikit-learn cannot perform inference on a GPU, and models like SVMs have a lot to gain from fast GPU primitives. Converting these models to PyTorch gives immediate access to these primitives.
 2. While scikit-learn supports serialization through pickle, saved models [are not reproducible](https://scikit-learn.org/stable/modules/model_persistence.html) across versions of the library. On the other hand, TorchScript provides a convenient, safe way to save a model with its corresponding implementation.
 3. While certain models like SVMs and linear classifiers are theoretically end-to-end differentiable, scikit-learn provides no mechanism to compute gradients through trained models. PyTorch provides this functionality mostly for free.

# Usage

First, train a model with scikit-learn as usual:

```python
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

x, y = create_some_dataset()
model = Pipeline([
    ("center", StandardScaler(with_std=False)),
    ("classify", SGDClassifier()),
])
model.fit(x, y)
```

Then call `sk2torch.wrap` on the model to create a PyTorch equivalent:

```python
import sk2torch
import torch

torch_model = sk2torch.wrap(model)
print(torch_model.predict(torch.tensor([[1., 2., 3.]]).double()))
```

You can save a model with TorchScript:

```python
import torch.jit

torch.jit.script(torch_model).save("path.pt")
# ...
loaded_model = torch.jit.load("path.pt")
```
