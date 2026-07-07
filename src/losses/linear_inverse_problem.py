import torch
import torch.nn.functional as F
from torch import nn


class LinearInverseProblemLoss(nn.Module):
    def __init__(self, operator, dist="l2_norm"):
        super().__init__()
        self.operator = operator
        self.dist = dist

    def forward(self, x, y):
        if self.dist == "l2_norm":
            return torch.linalg.norm(self.operator(x) - y)
        elif self.dist == "mse":
            return F.mse_loss(self.operator(x), y)
