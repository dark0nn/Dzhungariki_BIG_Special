import torch
from .base import BaseLoss

class CrosstalkLoss(BaseLoss):
    def forward(self, preds: list, targets: list) -> torch.Tensor:
        N = len(preds)
        return 0.1 * sum(torch.mean(preds[j]) for i in range(N) for j in range(N) if i != j) / (N * (N - 1))
