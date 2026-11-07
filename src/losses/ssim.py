import torch
import torch.nn.functional as F
from .base import BaseLoss

class SSIMLoss(BaseLoss):
    def forward(self, preds: list, targets: list) -> torch.Tensor:
        total = 0.0
        for pred, target in zip(preds, targets):
            p, t = pred / (pred.max() + 1e-8), target / (target.max() + 1e-8)
            mu_p, mu_t = F.avg_pool2d(p, 3, 1, 1), F.avg_pool2d(t, 3, 1, 1)
            var_p = F.avg_pool2d(p**2, 3, 1, 1) - mu_p**2
            var_t = F.avg_pool2d(t**2, 3, 1, 1) - mu_t**2
            cov = F.avg_pool2d(p * t, 3, 1, 1) - mu_p * mu_t
            ssim = ((2*mu_p*mu_t + 0.0001) * (2*cov + 0.0009)) / ((mu_p**2 + mu_t**2 + 0.0001) * (var_p + var_t + 0.0009))
            total += ssim.mean()
        return 1.0 - total / len(preds)
