import torch
import torch.nn as nn
from .propagation import FresnelPropagator
from .doe import PhaseDOE

class OpticalSystem(nn.Module):
    def __init__(self, distances: list, wavelength: float = 532e-9,
                 pixel_size: float = 8e-6, image_size: int = 128):
        super().__init__()
        self.distances = distances
        self.doe = PhaseDOE(image_size=image_size)
        self.propagators = nn.ModuleList([
            FresnelPropagator(dist, wavelength, pixel_size, image_size)
            for dist in distances
        ])
    
    def forward(self, phase: torch.Tensor) -> list:
        field = self.doe(phase.squeeze(0).squeeze(0))
        reconstructions = []
        for prop in self.propagators:
            intensity = torch.abs(prop(field))**2
            reconstructions.append((intensity / (intensity.max() + 1e-8)).unsqueeze(0).unsqueeze(0))
        return reconstructions
