import torch
import torch.nn.functional as F
from .base import BaseLoss

class EdgeLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        self.sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
    def forward(self, preds: list, targets: list) -> torch.Tensor:
        total = 0.0
        for pred, target in zip(preds, targets):
            sx, sy = self.sx.to(pred.device), self.sy.to(pred.device)
            gp, gt = torch.sqrt(F.conv2d(pred,sx,padding=1)**2 + F.conv2d(pred,sy,padding=1)**2), \
                     torch.sqrt(F.conv2d(target,sx,padding=1)**2 + F.conv2d(target,sy,padding=1)**2)
            total += F.l1_loss(gp, gt)
        return total / len(preds)
