"""
Train an SVM on a 2D classification problem, then plot a gradient vector field
for the predicted class probabilty.
"""

import matplotlib.pyplot as plt
import numpy as np
import sk2torch
import torch
from sklearn.svm import SVC

# Create a dataset of two Gaussians. There will be some overlap
# between the two classes, which adds some uncertainty to the model.
xs = np.concatenate(
    [
        np.random.random(size=(256, 2)) + [1, 0],
        np.random.random(size=(256, 2)) + [-1, 0],
    ],
    axis=0,
)
ys = np.array([False] * 256 + [True] * 256)

# Train an SVM on the data and wrap it in PyTorch.
sk_model = SVC(probability=True)
sk_model.fit(xs, ys)
model = sk2torch.wrap(sk_model)

# Create a coordinate grid to compute a vector field on.
spaced = np.linspace(-2, 2, num=25)
grid_xs = torch.tensor([[x, y] for x in spaced for y in spaced], requires_grad=True)

# Compute the gradients of the SVM output.
outputs = model.predict_proba(grid_xs)[:, 1]
(input_grads,) = torch.autograd.grad(outputs.sum(), (grid_xs,))

# Create a quiver plot of the vector field.
plt.quiver(
    grid_xs[:, 0].detach().numpy(),
    grid_xs[:, 1].detach().numpy(),
    input_grads[:, 0].detach().numpy(),
    input_grads[:, 1].detach().numpy(),
)
plt.savefig("svm_vector_field.png")
