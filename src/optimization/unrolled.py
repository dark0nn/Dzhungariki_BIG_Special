import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class UnrolledHolography(nn.Module):
    def __init__(self, num_iterations: int = 20, initial_lr: float = 0.01):
        super().__init__()
        self.num_iterations = num_iterations
        self.learning_rates = nn.Parameter(torch.ones(num_iterations) * initial_lr, requires_grad=True)
    
    def forward(self, targets, system, loss_fn, initial_phase=None):
        device = next(system.parameters()).device
        phase = torch.rand(1, 1, 128, 128, device=device, requires_grad=True) * 2 * np.pi - np.pi
        
        for i in range(self.num_iterations):
            reconstructions = system(phase)
            loss, _ = loss_fn(reconstructions, targets)
            grad = torch.autograd.grad(loss, phase, create_graph=True, retain_graph=(i < self.num_iterations - 1))[0]
            phase = phase - self.learning_rates[i] * grad
            phase = torch.atan2(torch.sin(phase), torch.cos(phase)) * np.pi
        return phase

class UnrolledOptimizer:
    def __init__(self, num_iterations: int = 20, initial_lr: float = 0.01,
                 training_epochs: int = 50, training_lr: float = 0.001):
        self.model = UnrolledHolography(num_iterations, initial_lr)
        self.training_epochs = training_epochs
        self.training_lr = training_lr
        self._trained = False
    
    def train(self, dataset, system, loss_fn):
        self.model = self.model.to(next(system.parameters()).device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.training_lr)
        for epoch in range(self.training_epochs):
            for targets, distances in dataset:
                targets = [t.to(next(system.parameters()).device) for t in targets]
                optimizer.zero_grad()
                phase = self.model(targets, system, loss_fn)
                final_loss, _ = loss_fn(system(phase), targets)
                final_loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{self.training_epochs}, Loss: {final_loss.item():.6f}")
        self._trained = True

    def optimize(self, targets, distances, system, loss_fn):
        self.model = self.model.to(next(system.parameters()).device)
        targets = [t.to(next(system.parameters()).device) for t in targets]
        with torch.no_grad():
            return self.model(targets, system, loss_fn)
