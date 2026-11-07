import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from src.utils import load_config, set_seed
from src.data.dataset import MultiPlaneDataset
from src.physics.system import OpticalSystem
from src.losses.base import CompositeLoss
from src.losses.ssim import SSIMLoss
from src.losses.pcc import PCCLoss
from src.losses.edge import EdgeLoss
from src.losses.crosstalk import CrosstalkLoss
from src.optimization.unrolled import UnrolledOptimizer
from src.metrics.evaluator import Evaluator
from src.visualization.plots import plot_results

def main():
    cfg = load_config('configs/unrolled.yaml')
    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = MultiPlaneDataset(root=cfg['data']['root'], image_size=cfg['data']['image_size'])
    system = OpticalSystem(distances=cfg['distances'], wavelength=cfg['wavelength'],
                           pixel_size=cfg['pixel_size'], image_size=cfg['image_size']).to(device)
    
    loss_fn = CompositeLoss()
    for name, weight in cfg.get('losses', {}).items():
        if name == 'ssim': loss_fn.add_loss('ssim', SSIMLoss(), weight)
        elif name == 'pcc': loss_fn.add_loss('pcc', PCCLoss(), weight)
        elif name == 'edge': loss_fn.add_loss('edge', EdgeLoss(), weight)
        elif name == 'crosstalk': loss_fn.add_loss('crosstalk', CrosstalkLoss(), weight)
    
    optimizer = UnrolledOptimizer(num_iterations=cfg['unrolled']['num_iterations'],
                                  initial_lr=cfg['unrolled']['initial_lr'],
                                  training_epochs=cfg['training']['epochs'],
                                  training_lr=cfg['training']['lr'])
    
    print("Training...")
    optimizer.train(dataset, system, loss_fn)
    
    targets, distances = dataset[0]
    targets = [t.to(device) for t in targets]
    phase = optimizer.optimize(targets, distances, system, loss_fn)
    reconstructions = system(phase)
    
    metrics = Evaluator.evaluate_all(reconstructions, targets)
    print(f"\nResults: PCC Mean = {metrics['pcc_mean']:.4f}")
    
    plot_results(targets, reconstructions, phase, save_path='experiments/results/optimization.png')
    print("Saved to experiments/results/optimization.png")

if __name__ == '__main__':
    main()
