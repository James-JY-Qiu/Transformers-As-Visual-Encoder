import torch
from torch import nn


class ReZero(nn.Module):
    def __init__(self, fn):
        """Introduce a learnable scaling parameter to modulate the output of a function.

        :param fn: The function or layer
        """
        super().__init__()
        # initialize the trainable scaling parameter
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class PreScaleNorm(nn.Module):
    def __init__(self, fn, eps=1e-5):
        """Introduce a learnable scaling parameter to normalize the input tensor before applying the function.

        :param fn: The function or layer
        :param eps: A small number to add to the denominator for numerical stability
        """
        super().__init__()
        self.fn = fn
        # initialize the trainable scaling parameter
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        # calculate the norm of the input tensor
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        # normalize the input tensor and scale it
        x = x / n * self.g
        return self.fn(x, **kwargs)


class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        """Introduce a layer normalization layer before applying the function.

        :param dim: The feature dimension
        :param fn: The function or layer
        """
        super().__init__()
        # apply layer normalization to the input tensor
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)