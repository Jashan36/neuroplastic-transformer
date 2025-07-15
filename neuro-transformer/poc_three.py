import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
# from datasets import load_dataset  # Removed since we're not using Hugging Face datasets
import nltk
from torch.cuda.amp import autocast, GradScaler
import random
import os
import json
import csv  # Added for local dataset loading

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Configuration
target_length = 50
plasticity_lr = 1e-6
gamma = 0.95
beta = 1.0
lambda_decay = 0.01
lambda_reward = 0.2  # Initial reward weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loss weights
alpha = 5.0  # Efficiency
beta_sem = 0.7  # Semantic
gamma_div = 0.3  # Diversity
delta_ref = 0.4  # Reference-based similarity

# Curriculum stages
CURRICULUM = [
    {"name": "definitions", "max_length": 30, "epochs": 1},
    {"name": "explanations", "max_length": 80, "epochs": 2},
    {"name": "dialogues", "max_length": 150, "epochs": 1}
]

# Initialize model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
tokenizer.pad_token = tokenizer.eos_token

# Plasticity Wrapper with stabilization enhancements
class NeuroPlasticWrapper(nn.Module):
    def __init__(self, model, gamma, beta, lambda_decay):
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.beta = beta
        self.lambda_decay = lambda_decay
        
        # Initialize eligibility traces
        self.eligibility_traces = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and "ln" not in name and "embed" not in name:
                self.eligibility_traces[name] = torch.zeros_like(param.data)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def update_eligibility_traces(self):
        """Update traces with normalized absolute gradients"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.eligibility_traces and param.grad is not None:
                    grad = param.grad.detach()
                    # Normalize and clip gradients before adding to traces
                    grad_norm = torch.norm(grad)
                    normalized_grad = grad / (grad_norm + 1e-8)
                    self.eligibility_traces[name] = (
                        self.gamma * self.eligibility_traces[name] + 
                        torch.abs(normalized_grad)
                    )
    
    def plastic_update(self, reward_signals):
        """Apply stabilized neuroplastic weight update"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.eligibility_traces:
                    # Get reward signal for this parameter
                    reward = reward_signals.get(name, 0)
                    
                    # Clamp eligibility traces to prevent explosion
                    trace = torch.clamp(self.eligibility_traces[name], min=-1, max=1)
                    
                    # Apply update with weight decay
                    plastic_update = plasticity_lr * (
                        self.beta * reward * trace -
                        self.lambda_decay * param.data
                    )
                    param.data += plastic_update

# Initialize wrapper and optimizer
wrapper = NeuroPlasticWrapper(model, gamma, beta, lambda_decay).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Initialize GradScaler only if CUDA is available
if device.type == 'cuda':
    scaler = GradScaler()
else:
    scaler = None
    print("CUDA not available - disabling mixed precision training")

# Reward calculation functions
def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate"""
    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())
    return nltk.translate.bleu_score.sentence_bleu([ref_tokens], cand_tokens)

def calculate_meteor(reference, candidate):
    """Calculate METEOR score between reference and candidate"""
    return nltk.translate.meteor_score.meteor_score([reference], candidate)

# Enhanced dataset using local data
class CurriculumDataset(Dataset):
    def __init__(self, stage="definitions"):
        self.examples = []
        
        # Load from local CSV file
        try:
            with open('dolly_data.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for item in reader:
                    if stage == "definitions" and len(item['response']) < 100:
                        self.examples.append({
                            "instruction": f"Define: {item['instruction']}",
                            "response": item['response']
                        })
                    elif stage == "explanations" and len(item['response']) >= 100:
                        self.examples.append({
                            "instruction": item['instruction'],
                            "response": item['response']
                        })
                    elif stage == "dialogues":
                        self.examples.append({
                            "instruction": f"Explain in a dialogue: {item['instruction']}",
                            "response": item['response']
                        })
            print(f"Loaded {len(self.examples)} examples from local file for stage: {stage}")
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            # Fallback to synthetic data
            self.examples = self.generate_synthetic_examples(stage, 1000)
            print(f"Generated {len(self.examples)} synthetic examples for stage: {stage}")
    
    def generate_synthetic_examples(self, stage, count):
        """Generate synthetic training examples"""
        examples = []
        topics = [
            "machine learning", "renaissance art", "quantum physics",
            "ancient history", "software engineering", "literary analysis"
        ]
        
        for i in range(count):
            topic = random.choice(topics)
            if stage == "definitions":
                examples.append({
                    "instruction": f"Define: {topic}",
                    "response": f"{topic} is the study of..." 
                })
            elif stage == "explanations":
                examples.append({
                    "instruction": f"Explain: {topic}",
                    "response": f"{topic} involves several key concepts. First..." 
                })
            else:  # dialogues
                examples.append({
                    "instruction": f"Discuss: {topic}",
                    "response": f"Person A: What is {topic}?\nPerson B: It's a field that..." 
                })
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]["instruction"], self.examples[idx]["response"]

# Create datasets for each curriculum stage
datasets = {}
for stage in set(s["name"] for s in CURRICULUM):
    datasets[stage] = CurriculumDataset(stage)

# Training function with mixed precision
def train_step(batch, curriculum_stage, lambda_reward):
    instructions, responses = batch
    
    # Tokenize with curriculum-based max length
    max_length = next(s for s in CURRICULUM if s["name"] == curriculum_stage)["max_length"]
    inputs = tokenizer(
        instructions, 
        padding=True, 
        truncation=True, 
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    # Prepare labels for CE loss
    response_inputs = tokenizer(
        responses, 
        padding=True, 
        truncation=True, 
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    # Conditionally use autocast if CUDA available
    if device.type == 'cuda':
        with autocast():
            # --- STANDARD LANGUAGE MODELING ---
            outputs = model(**inputs, labels=response_inputs.input_ids)
            loss_ce = outputs.loss
            
            # --- GENERATE RESPONSE FOR REWARD CALCULATION ---
            gen_outputs = model.generate(
                inputs.input_ids,
                max_length=max_length + target_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                repetition_penalty=1.2
            )
            gen_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            
            # Differentiable forward pass on generated sequences
            model_outputs = model(gen_outputs, output_hidden_states=True)
            hidden_states = model_outputs.hidden_states[-1]
            
            # Initialize reward components
            eff_loss = torch.tensor(0.0, device=device)
            sem_loss = torch.tensor(0.0, device=device)
            div_loss = torch.tensor(0.0, device=device)
            ref_loss = torch.tensor(0.0, device=device)
            
            # Calculate rewards per item in batch
            for i in range(len(gen_texts)):
                prompt_length = inputs.input_ids[i].shape[0]
                gen_response = gen_texts[i][len(instructions[i]):].strip()
                output_length = len(tokenizer.encode(gen_response))
                
                # 1. Efficiency Loss
                length_diff = output_length - target_length
                item_eff = F.softplus(torch.tensor(length_diff, device=device))
                eff_loss += torch.clamp(item_eff, max=20.0)
                
                # 2. Semantic Loss
                prompt_embed = hidden_states[i, :prompt_length].mean(dim=0)
                gen_embed = hidden_states[i, prompt_length:].mean(dim=0)
                cos_sim = F.cosine_similarity(prompt_embed.unsqueeze(0), 
                                             gen_embed.unsqueeze(0), dim=1)
                sem_loss += 1.0 - cos_sim
                
                # 3. Diversity Loss
                if prompt_length < gen_outputs.shape[1] - 1:
                    gen_logits = model_outputs.logits[i, prompt_length-1:-1]
                    probs = F.softmax(gen_logits, dim=-1)
                    log_probs = F.log_softmax(gen_logits, dim=-1)
                    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
                    div_loss += -entropy
                else:
                    div_loss += torch.tensor(0.0, device=device)
                
                # 4. Reference-based Loss (BLEU + METEOR)
                try:
                    bleu_score = calculate_bleu(responses[i], gen_response)
                    meteor_score_val = calculate_meteor(responses[i], gen_response)
                    ref_loss += 1.0 - (0.5 * bleu_score + 0.5 * meteor_score_val)
                except:
                    ref_loss += torch.tensor(1.0, device=device)  # Max penalty if error
            
            # Average losses across batch
            batch_size = len(instructions)
            eff_loss /= batch_size
            sem_loss /= batch_size
            div_loss /= batch_size
            ref_loss /= batch_size
            
            # Combined reward loss
            reward_loss = (
                alpha * eff_loss + 
                beta_sem * sem_loss + 
                gamma_div * div_loss +
                delta_ref * ref_loss
            )
            
            # Combined total loss
            total_loss = loss_ce + lambda_reward * reward_loss
    else:
        # CPU-only path
        outputs = model(**inputs, labels=response_inputs.input_ids)
        loss_ce = outputs.loss
        
        # Generate responses without autocast
        gen_outputs = model.generate(
            inputs.input_ids,
            max_length=max_length + target_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            repetition_penalty=1.2
        )
        gen_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
        
        # Rest of reward calculation same as above
        # ... [same reward calculation code as in CUDA block] ...
        
        # Combined total loss
        total_loss = loss_ce + lambda_reward * reward_loss
    
    # Backpropagate with gradient clipping
    if scaler:
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    optimizer.zero_grad()
    
    # Prepare plasticity update signals
    reward_signals = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in wrapper.eligibility_traces and param.grad is not None:
                reward_signals[name] = -param.grad.detach()
    
    # Apply neuroplastic update
    wrapper.plastic_update(reward_signals)
    
    return {
        "loss_ce": loss_ce.item(),
        "reward_loss": reward_loss.item(),
        "efficiency": eff_loss.item(),
        "semantic": sem_loss.item(),
        "diversity": -div_loss.item(),
        "reference": ref_loss.item()
    }

# ... [Rest of the code remains the same] ...
