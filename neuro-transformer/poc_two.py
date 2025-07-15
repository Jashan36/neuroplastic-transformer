import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from torch.optim import AdamW
import numpy as np

# Configuration
target_length = 50  # Target output length
plasticity_lr = 1e-5  # Plasticity learning rate
gamma = 0.95  # Eligibility trace decay
beta = 1.0  # Reward weighting (1.0 = full reward-driven learning)
lambda_decay = 0.01  # Weight decay
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights
alpha = 1.0  # Efficiency loss weight
beta_sem = 0.5  # Semantic loss weight
gamma_div = 0.3  # Diversity loss weight

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
        """Update traces using absolute gradients"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.eligibility_traces and param.grad is not None:
                    self.eligibility_traces[name] = (
                        self.gamma * self.eligibility_traces[name] + 
                        torch.abs(param.grad.detach())
                    )
    
    def plastic_update(self, reward_signals):
        """Apply neuroplastic weight update"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.eligibility_traces:
                    # Master equation: δₖMᵢⱼ,ₖ term
                    reward_term = self.beta * reward_signals.get(name, 0) * self.eligibility_traces[name]
                    
                    # Apply update with weight decay
                    param.data += plasticity_lr * (
                        reward_term - 
                        self.lambda_decay * param.data
                    )

# Initialize wrapper
wrapper = NeuroPlasticWrapper(model, gamma, beta, lambda_decay).to(device)
optimizer = AdamW(model.parameters(), lr=0)  # Dummy optimizer

# Training Loop
prompt = "Explain photosynthesis"
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
prompt_length = prompt_ids.size(1)

for iteration in range(10):  # 10 generations
    try:
        # Generate output (detached from graph)
        with torch.no_grad():
            outputs = model.generate(
                prompt_ids, 
                max_length=200, 
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_length = outputs.size(1) - prompt_length
        
        # Get the generated text (without prompt)
        generated_text = output_text[len(prompt):].strip()
        
        # Forward pass through model to get differentiable outputs
        model.zero_grad()
        model_output = model(outputs, output_hidden_states=True)
        logits = model_output.logits
        hidden_states = model_output.hidden_states[-1]  # Last layer hidden states
        
        # 1. Efficiency Loss
        loss_efficiency = F.softplus(
            torch.tensor(output_length - target_length, dtype=torch.float32, device=device)
        )
        
        # 2. Semantic Similarity Loss
        # Get embeddings for prompt and generated text
        prompt_embed = hidden_states[0, :prompt_length].mean(dim=0)
        generated_embed = hidden_states[0, prompt_length:].mean(dim=0)
        
        # Cosine similarity loss (1 - cosθ)
        cos_sim = F.cosine_similarity(prompt_embed.unsqueeze(0), 
                                     generated_embed.unsqueeze(0), 
                                     dim=1)
        loss_semantic = 1.0 - cos_sim
        
        # 3. Diversity Loss (using token entropy)
        # Focus on generated tokens only
        gen_logits = logits[0, prompt_length-1:-1]  # Logits for generated tokens
        
        # Compute token probabilities
        probs = F.softmax(gen_logits, dim=-1)
        log_probs = F.log_softmax(gen_logits, dim=-1)
        
        # Entropy: H = -Σ p * log(p)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        loss_diversity = -entropy  # Minimizing this maximizes entropy
        
        # Combined Total Loss
        total_loss = (
            alpha * loss_efficiency + 
            beta_sem * loss_semantic + 
            gamma_div * loss_diversity
        )
        
        # Backpropagate to get reward signal (δₖ = -∇W Loss)
        optimizer.zero_grad()
        total_loss.backward()
        
        # Update eligibility traces
        wrapper.update_eligibility_traces()
        
        # Compute reward signals for each parameter
        reward_signals = {}
        for name, param in model.named_parameters():
            if name in wrapper.eligibility_traces and param.grad is not None:
                reward_signals[name] = -param.grad.data
        
        # Apply neuroplastic update
        wrapper.plastic_update(reward_signals)
        
        # Monitoring
        print(f"\nIteration {iteration + 1}:")
        print(f"Output length: {output_length} | Total Loss: {total_loss.item():.2f}")
        print(f"Efficiency: {loss_efficiency.item():.2f} | Semantic: {loss_semantic.item():.2f} | Diversity: {-loss_diversity.item():.2f}")
        print("Output:", generated_text[:200] + "..." if len(generated_text) > 200 else generated_text)
    
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("OOM error, reducing memory usage")
            torch.cuda.empty_cache()
            # Add gradient checkpointing or reduce batch size here
        else:
            raise e
    except Exception as e:
        print(f"Error in iteration {iteration}: {str(e)}")
        break

# Test final behavior
print("\n=== Testing Final Model ===")
with torch.no_grad():
    final_output = model.generate(
        prompt_ids, 
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    final_text = tokenizer.decode(final_output[0], skip_special_tokens=True)
    final_length = final_output.size(1) - prompt_length
    print("Final Output Length:", final_length)
    print("Text:", final_text[len(prompt):].strip()[:200] + "...")