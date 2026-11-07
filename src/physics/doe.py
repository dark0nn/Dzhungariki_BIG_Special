import torch
import torch.nn as nn
import numpy as np

class PhaseDOE(nn.Module):
    def __init__(self, image_size: int = 128, num_bits: int = 8):
        super().__init__()
        self.num_levels = 2**num_bits
    
    def quantize(self, phase: torch.Tensor) -> torch.Tensor:
        phase_norm = torch.clamp((phase + np.pi) / (2 * np.pi), 0, 1)
        phase_quant = torch.round(phase_norm * (self.num_levels - 1)) / (self.num_levels - 1)
        phase_out = phase_norm + (phase_quant - phase_norm).detach()
        return phase_out * 2 * np.pi - np.pi
    
    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        return torch.exp(1j * self.quantize(phase))
