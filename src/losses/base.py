import torch
import torch.nn as nn
from abc import abstractmethod

class BaseLoss(nn.Module):
    @abstractmethod
    def forward(self, preds: list, targets: list) -> torch.Tensor: pass

class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = {}
    def add_loss(self, name: str, loss_fn: BaseLoss, weight: float = 1.0):
        self.losses[name] = (loss_fn, weight)
        self.add_module(name, loss_fn)
    def forward(self, preds: list, targets: list):
        total, details = 0.0, {}
        for name, (loss_fn, weight) in self.losses.items():
            value = loss_fn(preds, targets)
            details[name] = value.item()
            total = total + weight * value
        return total, details
