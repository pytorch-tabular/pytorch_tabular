import math

import pytest
import torch

from pytorch_tabular.models.common.layers.activations import entmax15, sparsemax


@pytest.mark.parametrize("activation", [sparsemax, entmax15])
@pytest.mark.parametrize("shape", [(4, 3), (2, 4, 3)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("k", [1, 2])
def test_partial_sort_negative_axis(activation, shape, axis, k):
    # Small, distinct logits keep all entries in the support, forcing top-k recursion.
    x = torch.linspace(-0.05, 0.05, steps=math.prod(shape), dtype=torch.double)
    x = x.reshape(shape).requires_grad_()
    expected = activation(x, dim=axis)
    actual = activation(x, dim=axis - x.ndim, k=k)

    torch.testing.assert_close(actual, expected)
    sums = actual.sum(dim=axis)
    torch.testing.assert_close(sums, torch.ones_like(sums))
    weights = torch.arange(x.numel(), dtype=x.dtype).reshape(shape)
    expected_grad = torch.autograd.grad((expected * weights).sum(), x)[0]
    actual_grad = torch.autograd.grad((actual * weights).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
