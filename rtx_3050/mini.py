import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class NeuroPlasticLite(nn.Module):
    """
    Hardware-Optimized NeuroPlastic Model for RTX 3050 6GB
    Specifications:
        - 256 neurons
        - 32 state dimensions
        - 16 feature dimensions
        - 4 batch size
        - 20 time steps
    """
    
    def __init__(self):
        super().__init__()
        self.N = 256      # Number of neurons
        self.d = 32       # State dimension
        self.k = 16       # Feature dimension
        self.k_neighbors = 50  # Top-k sparse attention
        
        # Biological parameters
        self.gamma = 0.1   # Leak rate
        self.tau = 1.0     # Distance decay
        self.eta = 0.01    # Learning rate
        self.lambda_w = 0.001  # Weight decay
        self.lambda_a = 0.95   # Activity smoothing
        
        # Neural states
        self.register_buffer('x', torch.zeros(4, self.N, self.d))
        self.register_buffer('a_bar', torch.zeros(4, self.N))
        
        # Learnable parameters
        self.features = nn.Parameter(torch.randn(self.N, self.k))
        self.bias = nn.Parameter(torch.zeros(self.N, self.d))
        self.W = nn.Parameter(torch.randn(self.N, self.N) * 0.1)
        
        # Input projection
        self.input_proj = nn.Linear(128, self.d)  # Assumes 128-dim input
        
        # Nonlinear functions
        self.phi = nn.Tanh()  # Activation
        self.sigma = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, self.d)
        )

    def compute_sparse_similarity(self) -> Tensor:
        """Top-k sparse neighborhood attention"""
        # Compute cosine similarity
        sim = torch.mm(F.normalize(self.features), F.normalize(self.features).t())
        
        # Get top-k neighbors for each neuron
        topk_val, topk_idx = torch.topk(sim, self.k_neighbors, dim=1)
        
        # Create sparse matrix
        row = torch.arange(self.N).repeat_interleave(self.k_neighbors)
        col = topk_idx.flatten()
        values = topk_val.flatten()
        
        return torch.sparse_coo_tensor(
            torch.stack([row, col]), 
            values,
            size=(self.N, self.N)
        )

    def compute_effective_weights(self) -> Tensor:
        """Sparse synaptic fading"""
        D = torch.cdist(self.features, self.features, p=2)
        return self.W * torch.exp(-D / self.tau)

    def forward(self, u: Tensor, dt: float = 0.05) -> Tensor:
        """Forward pass with memory optimizations"""
        # 1. Get sparse attention matrix
        N_sparse = self.compute_sparse_similarity()
        
        # 2. Compute effective weights
        W_eff = self.compute_effective_weights()
        
        # 3. Time-stepping loop
        for _ in range(20):  # 20 time steps
            # 4. Compute activations
            a = self.phi(torch.norm(self.x, p=2, dim=-1))
            
            # 5. Sparse synaptic input
            synaptic_input = torch.sparse.mm(N_sparse, a)
            
            # 6. Neuron state dynamics
            sigma_out = self.sigma(synaptic_input.unsqueeze(-1))
            u_proj = self.input_proj(u)
            dx = -self.gamma * self.x + sigma_out + self.bias + u_proj
            
            # 7. Update state
            self.x = self.x + dx * dt
            
            # 8. Update activity
            a_new = self.phi(torch.norm(self.x, p=2, dim=-1))
            self.a_bar = self.lambda_a * self.a_bar + (1 - self.lambda_a) * a_new
            
            # 9. Sparse plasticity update
            if self.training:
                co_activity = torch.bmm(
                    self.a_bar.unsqueeze(2), 
                    self.a_bar.unsqueeze(1)
                ).mean(0)
                
                # Sparse update
                dW = self.eta * torch.sparse.mm(N_sparse, co_activity) - self.lambda_w * self.W
                self.W += dW * dt
        
        return self.x

# Optimization Wrapper
def optimize_model(model):
    """Applies RTX 3050 optimizations"""
    model = model.half()  # FP16 precision
    model = torch.compile(model)  # PyTorch 2.0+ compilation
    
    # Enable automatic mixed precision
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    
    return model

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = NeuroPlasticLite().to(device)
    model = optimize_model(model)
    
    # Print memory summary
    print(f"Model VRAM: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
    
    # Sample input (batch size 4, input dim 128)
    u = torch.randn(4, 128, device=device).half()  # FP16 input
    
    # Warm-up
    for _ in range(3):
        _ = model(u)
    
    # Benchmark
    import time
    start = time.time()
    states = model(u)
    print(f"Inference time: {(time.time()-start)*1000:.2f} ms")
    
    print(f"Output states shape: {states.shape}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(device)/1e6:.2f} MB")
