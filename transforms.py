import random

import torch
from torch import nn


class RandomGammaCorrection(nn.Module):
    def __init__(self, vmin: float = 0.5, vmax: float = 2.0) -> None:
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        gamma: float = random.random() * (self.vmax - self.vmin) + self.vmin
        return img.pow(gamma)
