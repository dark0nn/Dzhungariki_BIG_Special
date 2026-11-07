import torch
from .base import BaseLoss

class PCCLoss(BaseLoss):
    def forward(self, preds: list, targets: list) -> torch.Tensor:
        total_pcc = 0.0
        for pred, target in zip(preds, targets):
            p, t = pred.flatten() - pred.mean(), target.flatten() - target.mean()
            total_pcc += (p * t).sum() / (torch.sqrt((p**2).sum() * (t**2).sum()) + 1e-8)
        return 1.0 - total_pcc / len(preds)
