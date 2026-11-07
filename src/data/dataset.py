import torch
import numpy as np
from pathlib import Path

class MultiPlaneDataset:
    def __init__(self, root: str, image_size: int = 128, num_planes: int = 4):
        self.root = Path(root)
        self.image_size = image_size
        self.num_planes = num_planes
        self.samples = []
        self._load_or_generate()
    
    def _load_or_generate(self):
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                sample_dir = self.root / f"sample_{i:04d}"
                sample_dir.mkdir(exist_ok=True)
                for j in range(self.num_planes):
                    x = np.linspace(-1, 1, self.image_size)
                    X, Y = np.meshgrid(x, x)
                    img = (X**2 + Y**2 < 0.3).astype(np.float32) if j==0 else (np.abs(X)+np.abs(Y)<0.5).astype(np.float32)
                    np.save(sample_dir / f"target_{j}.npy", img)
                self.samples.append(sample_dir)
        else:
            self.samples = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        targets = [torch.from_numpy(np.load(self.samples[idx] / f"target_{j}.npy")).unsqueeze(0).unsqueeze(0).float() for j in range(self.num_planes)]
        return targets, [0.1, 0.2, 0.3, 0.4]
