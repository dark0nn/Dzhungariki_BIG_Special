import torch
import torch.nn as nn
import numpy as np

class FresnelPropagator(nn.Module):
    def __init__(self, distance: float, wavelength: float = 532e-9, 
                 pixel_size: float = 8e-6, image_size: int = 128):
        super().__init__()
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.image_size = image_size
        self._precompute()
    
    def _precompute(self):
        N = self.image_size
        x = torch.linspace(-N//2, N//2-1, N) * self.pixel_size
        y = torch.linspace(-N//2, N//2-1, N) * self.pixel_size
        X, Y = torch.meshgrid(x, y, indexing='ij')
        R2 = X**2 + Y**2
        k = 2 * np.pi / self.wavelength
        phase = k / (2 * self.distance) * R2
        quad_phase = torch.exp(1j * phase)
        self.register_buffer('quad_phase', quad_phase)
        const = np.exp(1j * k * self.distance) / (1j * self.wavelength * self.distance)
        self.register_buffer('const', torch.tensor(const))
    
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field_mod = field * self.quad_phase
        field_fft = torch.fft.fftshift(torch.fft.fft2(field_mod))
        return self.const * field_fft * self.quad_phase
