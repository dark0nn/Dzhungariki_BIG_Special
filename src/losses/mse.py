import torch.nn.functional as F
from .base import BaseLoss

class MSELoss(BaseLoss):
    def forward(self, preds: list, targets: list) -> torch.Tensor:
        return sum(F.mse_loss(p, t) for p, t in zip(preds, targets)) / len(preds)
