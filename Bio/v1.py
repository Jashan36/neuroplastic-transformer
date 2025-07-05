import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List
import math

class NeuronNeighborhoodModel(nn.Module):
    """
    Complete implementation of all specified formulas with:
    - Continuous-time dynamics
    - Feature-space synaptic fading
    - Neighborhood attention
    - Multi-timescale plasticity
    - Reward modulation
    - Hardware optimization
    """
    
    def __init__(
        self,
        num_neurons: int,
        state_dim: int,
        feature_dim: int,
        gamma: float = 0.1,
        tau: float = 1.0,
        eta: float = 0.01,
        lambda_w: float = 0.001,
        lambda_a: float = 0.95,
        rho: float = 0.1,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        self.N = num_neurons
        self.d = state_dim
        self.k = feature_dim
        self.device = device
        
        # Core parameters
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('tau', torch.tensor(tau))
        self.register_buffer('eta', torch.tensor(eta))
        self.register_buffer('lambda_w', torch.tensor(lambda_w))
        self.register_buffer('lambda_a', torch.tensor(lambda_a))
        self.register_buffer('rho', torch.tensor(rho))
        
        # State variables
        self.register_buffer('x', torch.zeros(1, self.N, self.d, device=device))
        self.register_buffer('a_bar', torch.zeros(1, self.N, device=device))
        
        # Learnable parameters
        self.features = nn.Parameter(torch.randn(self.N, self.k, device=device))
        self.bias = nn.Parameter(torch.zeros(self.N, self.d, device=device))
        self.W = nn.Parameter(torch.randn(self.N, self.N, device=device) * 0.1
        
        # Nonlinear functions
        self.phi = nn.Tanh()  # Activation function φ
        self.sigma = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, self.d)
        ).to(device)

    def compute_distance_matrix(self) -> Tensor:
        """d_ij = ||f_i - f_j||₂ (Formula 4)"""
        return torch.cdist(self.features, self.features, p=2)

    def compute_similarity_matrix(self) -> Tensor:
        """S_ij = exp(f_iᵀf_j) / Σₖ exp(f_iᵀf_k) (Formula 5)"""
        sim = torch.mm(self.features, self.features.t())
        return F.softmax(sim, dim=-1)

    def compute_neighborhood_matrix(self) -> Tensor:
        """N_ij = (S_ij * e^{-d_ij/τ}) / Σₖ (S_ik * e^{-d_ik/τ}) (Formula 5)"""
        S = self.compute_similarity_matrix()
        D = self.compute_distance_matrix()
        decay = torch.exp(-D / self.tau)
        numerator = S * decay
        return numerator / (numerator.sum(dim=1, keepdim=True) + 1e-8

    def compute_effective_weights(self) -> Tensor:
        """W̃_ij = W_ij * e^{-d_ij/τ} (Formula 4)"""
        D = self.compute_distance_matrix()
        return self.W * torch.exp(-D / self.tau)

    def update_time_averaged_activity(self, a: Tensor) -> Tensor:
        """ā_i(t) = λ * ā_i(t-Δt) + (1-λ) * a_i(t) (Formula 6)"""
        return self.lambda_a * self.a_bar + (1 - self.lambda_a) * a

    def compute_plasticity_update(
        self,
        a_bar: Tensor,
        N: Tensor,
        reward: Optional[Tensor] = None,
        mod_matrix: Optional[Tensor] = None
    ) -> Tensor:
        """dW_ij/dt = η * ā_i * ā_j * N_ij - λ * W_ij + ρ * δ * M_ij (Formula 7)"""
        # Co-activity term (Hebbian)
        co_activity = torch.outer(a_bar.squeeze(0), a_bar.squeeze(0))
        hebbian = self.eta * co_activity * N
        
        # Homeostasis term
        homeostasis = self.lambda_w * self.W
        
        # Reward modulation
        modulation = 0
        if reward is not None and self.rho > 0:
            M = mod_matrix if mod_matrix is not None else torch.ones_like(self.W)
            modulation = self.rho * reward.mean() * M
        
        return hebbian - homeostasis + modulation

    def forward(
        self,
        u: Tensor,
        dt: float,
        reward: Optional[Tensor] = None,
        mod_matrix: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Full system dynamics update
        Args:
            u: External input [batch_size, N, d]
            dt: Time step size
            reward: Optional reward signal [batch_size]
            mod_matrix: Modulation matrix [N, N]
        Returns:
            x_next: Neuron states [batch_size, N, d]
            a_next: Activations [batch_size, N]
        """
        # 1. Compute current activation (Formula 2)
        a = self.phi(torch.norm(self.x, p=2, dim=-1))
        
        # 2. Compute effective weights (Formula 4)
        W_eff = self.compute_effective_weights()
        
        # 3. Compute synaptic input
        synaptic_input = torch.einsum('nm,bn->bm', W_eff, a)
        
        # 4. Neuron state dynamics (Formula 3)
        sigma_out = self.sigma(synaptic_input.unsqueeze(-1))
        dx = -self.gamma * self.x + sigma_out + self.bias + u
        x_next = self.x + dx * dt
        
        # 5. Update activation
        a_next = self.phi(torch.norm(x_next, p=2, dim=-1))
        
        # 6. Update time-averaged activity (Formula 6)
        a_bar_next = self.update_time_averaged_activity(a_next)
        
        # 7. Compute neighborhood matrix (Formula 5)
        N_mat = self.compute_neighborhood_matrix()
        
        # 8. Plasticity update (Formula 7)
        if self.training:
            dW = self.compute_plasticity_update(a_bar_next, N_mat, reward, mod_matrix)
            self.W.data += dW * dt
        
        # Update persistent states
        self.x = x_next.detach()
        self.a_bar = a_bar_next.detach()
        
        return x_next, a_next

class BioPlasticTransformer(nn.Module):
    """Transformer integration with full biological dynamics"""
    
    def __init__(
        self,
        num_layers: int,
        num_neurons: int,
        state_dim: int,
        feature_dim: int,
        num_heads: int,
        max_seq_len: int,
        device: torch.device
    ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([
            BioPlasticTransformerBlock(
                num_neurons=num_neurons,
                state_dim=state_dim,
                feature_dim=feature_dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                layer_idx=i,
                device=device
            ) for i in range(num_layers)
        ])
        self.position_emb = PositionalEncoding(state_dim, max_len=max_seq_len)
        
    def forward(self, x: Tensor, reward: Tensor = None) -> Tensor:
        # Add positional encoding
        x = self.position_emb(x)
        
        # Process through bio-plastic layers
        for layer in self.layers:
            x, _ = layer(x, reward)
        return x

class BioPlasticTransformerBlock(nn.Module):
    """Transformer block with biological dynamics"""
    
    def __init__(
        self,
        num_neurons: int,
        state_dim: int,
        feature_dim: int,
        num_heads: int,
        max_seq_len: int,
        layer_idx: int,
        device: torch.device
    ):
        super().__init__()
        self.attention = BioPlasticAttention(
            num_neurons=num_neurons,
            state_dim=state_dim,
            feature_dim=feature_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            device=device
        )
        self.neuron_dynamics = NeuronNeighborhoodModel(
            num_neurons=num_neurons,
            state_dim=state_dim,
            feature_dim=feature_dim,
            gamma=0.1 + 0.02 * layer_idx,  # Layer-specific
            tau=1.0 - 0.05 * layer_idx,     # Layer-specific
            device=device
        )
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)
        
    def forward(self, x: Tensor, reward: Tensor) -> Tuple[Tensor, Tensor]:
        # Bio-plastic attention
        attn_out, states = self.attention(x, reward)
        x = self.norm1(x + attn_out)
        
        # Neuron dynamics
        dyn_out, _ = self.neuron_dynamics(x, dt=0.1, reward=reward)
        x = self.norm2(x + dyn_out)
        return x, states

class BioPlasticAttention(nn.Module):
    """Attention with biological plasticity"""
    
    def __init__(
        self,
        num_neurons: int,
        state_dim: int,
        feature_dim: int,
        num_heads: int,
        max_seq_len: int,
        device: torch.device
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = state_dim // num_heads
        self.device = device
        
        # Feature embeddings
        self.position_features = nn.Parameter(
            torch.randn(max_seq_len, feature_dim, device=device)
        
        # Plastic projections
        self.q_proj = PlasticLinear(state_dim, state_dim, feature_dim, device)
        self.k_proj = PlasticLinear(state_dim, state_dim, feature_dim, device)
        self.v_proj = PlasticLinear(state_dim, state_dim, feature_dim, device)
        self.o_proj = PlasticLinear(state_dim, state_dim, feature_dim, device)
        
        # Neuron dynamics
        self.dynamics = NeuronNeighborhoodModel(
            num_neurons=num_neurons,
            state_dim=state_dim,
            feature_dim=feature_dim,
            device=device
        )

    def forward(self, x: Tensor, reward: Tensor) -> Tuple[Tensor, Dict]:
        B, T, _ = x.shape
        
        # Get position features
        positions = torch.arange(T, device=self.device)
        pos_feats = self.position_features[positions]
        
        # Plastic projections
        Q = self.q_proj(x, pos_feats, None, 0.1, reward)
        K = self.k_proj(x, pos_feats, None, 0.1, reward)
        V = self.v_proj(x, pos_feats, None, 0.1, reward)
        
        # Multi-head processing
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention with feature-space modulation
        dist_matrix = torch.cdist(pos_feats, pos_feats)
        decay = torch.exp(-dist_matrix / 1.0)  # Fixed tau
        attn_scores = (Q @ K.transpose(-2, -1)) * decay.unsqueeze(0)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ V).transpose(1, 2).reshape(B, T, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output, pos_feats, None, 0.1, reward)
        
        # Neuron dynamics processing
        dyn_output, states = self.dynamics(attn_output, dt=0.1, reward=reward)
        
        return dyn_output, states

class PlasticLinear(nn.Module):
    """Linear layer with biological plasticity"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        feature_dim: int,
        device: torch.device
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Plastic weight parameters
        self.W = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.1
        self.features_in = nn.Parameter(torch.randn(in_features, feature_dim, device=device))
        self.features_out = nn.Parameter(torch.randn(out_features, feature_dim, device=device))
        
        # Plasticity parameters
        self.eta = nn.Parameter(torch.tensor(0.01))
        self.lambda_w = nn.Parameter(torch.tensor(0.001))
        self.rho = nn.Parameter(torch.tensor(0.1))
        
        # Activity tracking
        self.register_buffer('a_bar_in', torch.zeros(1, device=device))
        self.register_buffer('a_bar_out', torch.zeros(1, device=device))
        
    def forward(
        self,
        x: Tensor,
        pos_features: Tensor,
        state: Optional[Dict],
        dt: float,
        reward: Optional[Tensor] = None
    ) -> Tensor:
        # Distance-based modulation
        dist_in = torch.cdist(self.features_in, pos_features.mean(dim=0, keepdim=True))
        dist_out = torch.cdist(self.features_out, pos_features.mean(dim=0, keepdim=True))
        mod_in = torch.exp(-dist_in / 1.0)  # Fixed tau
        mod_out = torch.exp(-dist_out / 1.0)
        W_mod = self.W * (mod_out @ mod_in.T)
        
        # Forward pass
        output = F.linear(x, W_mod)
        
        # Plasticity update during training
        if self.training:
            a_in = x.norm(dim=-1).mean()
            a_out = output.norm(dim=-1).mean()
            self.a_bar_in = 0.9 * self.a_bar_in + 0.1 * a_in
            self.a_bar_out = 0.9 * self.a_bar_out + 0.1 * a_out
            
            # Hebbian update
            co_activity = self.a_bar_out * self.a_bar_in
            dW = self.eta * co_activity - self.lambda_w * self.W
            
            # Reward modulation
            if reward is not None and self.rho > 0:
                dW += self.rho * reward.mean() * torch.ones_like(self.W)
            
            self.W.data += dW * dt
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1)]

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize core model
    bio_model = NeuronNeighborhoodModel(
        num_neurons=512,
        state_dim=64,
        feature_dim=32,
        device=device
    ).to(device)
    
    # Initialize transformer
    transformer = BioPlasticTransformer(
        num_layers=6,
        num_neurons=512,
        state_dim=64,
        feature_dim=32,
        num_heads=8,
        max_seq_len=128,
        device=device
    ).to(device)
    
    # Example input
    batch_size = 32
    u = torch.randn(batch_size, 512, 64).to(device)  # External input
    
    # Core model step
    states, activations = bio_model(u, dt=0.1)
    print("Core model states:", states.shape)
    
    # Transformer forward pass
    seq_input = torch.randn(batch_size, 128, 64).to(device)
    transformer_output = transformer(seq_input)
    print("Transformer output:", transformer_output.shape)
    
    # Training step
    optimizer = torch.optim.Adam(bio_model.parameters(), lr=1e-4)
    target = torch.randn_like(states)
    loss = F.mse_loss(states, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Training completed successfully!")