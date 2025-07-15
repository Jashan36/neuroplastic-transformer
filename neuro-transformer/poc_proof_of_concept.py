import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from torch.optim import AdamW

# Configuration
target_length = 50  # Target output length
plasticity_lr = 1e-5  # Plasticity learning rate
gamma = 0.95  # Eligibility trace decay
beta = 1.0  # Reward weighting (1.0 = full reward-driven learning)
lambda_decay = 0.01  # Weight decay
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
tokenizer.pad_token = tokenizer.eos_token

# Plasticity Wrapper
class NeuroPlasticWrapper(nn.Module):
    def __init__(self, model, gamma, beta, lambda_decay):
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.beta = beta
        self.lambda_decay = lambda_decay
        
        # Initialize eligibility traces for all linear layers
        self.eligibility_traces = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and "ln" not in name and "embed" not in name:
                self.eligibility_traces[name] = torch.zeros_like(param.data)
    
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
    
    def update_eligibility_traces(self):
        """Update traces using activations (simplified Hebbian update)"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.eligibility_traces:
                    # Simplified: Use gradient magnitudes as proxy for activation correlations
                    self.eligibility_traces[name] = (
                        self.gamma * self.eligibility_traces[name] + 
                        torch.abs(param.grad) if param.grad is not None else 0
                    )
    
    def plastic_update(self, reward_signals):
        """Apply neuroplastic weight update"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.eligibility_traces:
                    # Master equation: δₖMᵢⱼ,ₖ term
                    reward_term = self.beta * reward_signals[name] * self.eligibility_traces[name]
                    
                    # Apply update
                    param.data += plasticity_lr * (
                        reward_term - 
                        self.lambda_decay * param.data
                    )

# Initialize wrapper
wrapper = NeuroPlasticWrapper(model, gamma, beta, lambda_decay).to(device)
optimizer = AdamW(model.parameters(), lr=0)  # Dummy optimizer (actual updates via plasticity)

# Training Loop
prompt = "Explain photosynthesis"
for iteration in range(10):  # 10 generations
    # Generate output
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids, 
        max_length=200, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_length = len(outputs[0]) - input_ids.size(1)

    # Get logits for differentiable loss
    model_output = model(input_ids)
    logits = model_output.logits

    # Use input_ids as dummy target for cross-entropy
    target = input_ids.view(-1)
    loss_efficiency = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target
    )

    # Backpropagate to get reward signal (δₖ = -∇W Loss)
    optimizer.zero_grad()
    loss_efficiency.backward()
    
    # Update eligibility traces
    wrapper.update_eligibility_traces()
    
    # Compute reward signals for each parameter
    reward_signals = {}
    for name, param in model.named_parameters():
        if name in wrapper.eligibility_traces:
            reward_signals[name] = -param.grad.data if param.grad is not None else 0
    
    # Apply neuroplastic update
    wrapper.plastic_update(reward_signals)
    
    # Monitoring
    print(f"\nIteration {iteration + 1}:")
    print(f"Output length: {output_length} | Loss: {loss_efficiency.item():.2f}")
    print("Output:", output_text.replace(prompt, "")[:200] + "...")

# Test final behavior
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
final_output = model.generate(input_ids, max_length=200)
final_text = tokenizer.decode(final_output[0], skip_special_tokens=True)
print("\n=== Final Output ===")
print("Length:", len(final_output[0]) - input_ids.size(1))
print("Text:", final_text.replace(prompt, "")[:200] + "...")
