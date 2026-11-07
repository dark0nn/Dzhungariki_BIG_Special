import torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def plot_results(targets: list, reconstructions: list, phase: torch.Tensor, save_path: str = None):
    fig, axes = plt.subplots(2, len(targets) + 1, figsize=(15, 6))
    axes[0, 0].imshow(phase.squeeze().detach().cpu().numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0, 0].set_title('Phase Map'); axes[0, 0].axis('off')
    for i, (t, r) in enumerate(zip(targets, reconstructions)):
        axes[0, i+1].imshow(t.squeeze().detach().cpu().numpy(), cmap='gray'); axes[0, i+1].axis('off')
        axes[1, i+1].imshow(r.squeeze().detach().cpu().numpy(), cmap='gray'); axes[1, i+1].axis('off')
    axes[1, 0].axis('off')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
