import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 0.05):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.sigma * torch.randn_like(x, device=x.device)
        return x + noise


class PoissonNoise(nn.Module):
    def __init__(self, rate: float = 1.0):
        super().__init__()
        self.rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1)
        noisy = torch.poisson((x * 255.0 * self.rate))
        return (noisy / 255.0 / self.rate).clamp(0, 1)
