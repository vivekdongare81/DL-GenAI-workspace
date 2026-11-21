"""The model: an MLP that keeps its input normalization statistics as buffers."""

import torch
from torch import nn


class MLP(nn.Module):
    """28x28 image in, 10 class scores (logits) out."""

    def __init__(self, hidden: list[int], dropout: float = 0.0, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        layers, width = [], 28 * 28
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
            width = h
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x.flatten(1))
