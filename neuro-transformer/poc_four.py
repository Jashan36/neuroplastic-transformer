import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
import nltk
from torch.cuda.amp import autocast, GradScaler
import random
import os
import json

# Download required NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# Configuration - REDUCED FOR FASTER TESTING
target_length = 30
plasticity_lr = 1e-6
gamma = 0.95
beta = 1.0
lambda_decay = 0.01
lambda_reward = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loss weights
alpha = 5.0  # Efficiency
beta_sem = 0.7  # Semantic
gamma_div = 0.3  # Diversity
delta_ref = 0.4  # Reference-based similarity

# Curriculum stages - REDUCED EPOCHS
CURRICULUM = [
    {"name": "definitions", "max_length": 30, "epochs": 1},
    {"name": "explanations", "max_length": 60, "epochs": 1},
    {"name": "dialogues", "max_length": 100, "epochs": 1}
]

# Initialize model and tokenizer - USE SMALLER MODEL
print("Loading model...")
model_name = "gpt2"  # Use smaller model for faster testing
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
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
print("Initializing wrapper...")
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
    return nltk.translate.bleu_score.sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5))

def calculate_meteor(reference, candidate):
    """Calculate METEOR score between reference and candidate"""
    return nltk.translate.meteor_score.meteor_score([reference], candidate)

# Enhanced dataset using synthetic data only
class CurriculumDataset(Dataset):
    def __init__(self, stage="definitions"):
        self.examples = self.generate_synthetic_examples(stage, 100)  # Reduced dataset size
        print(f"Generated {len(self.examples)} synthetic examples for stage: {stage}")
    
    def generate_synthetic_examples(self, stage, count):
        """Generate synthetic training examples"""
        examples = []
        topics = [
            "machine learning", "renaissance art", "quantum physics",
            "ancient history", "software engineering", "literary analysis",
            "photosynthesis", "neural networks", "climate change",
            "economic theory", "human evolution", "quantum computing"
        ]
        
        for i in range(count):
            topic = random.choice(topics)
            if stage == "definitions":
                examples.append({
                    "instruction": f"Define: {topic}",
                    "response": f"{topic} is the study of {random.choice(['scientific', 'historical', 'technical'])} concepts related to {topic.split()[-1]}." 
                })
            elif stage == "explanations":
                examples.append({
                    "instruction": f"Explain: {topic}",
                    "response": f"{topic} involves several key concepts. First, {random.choice(['it deals with', 'it focuses on', 'it examines'])} {random.choice(['fundamental principles', 'core ideas', 'basic elements'])}. Second, {random.choice(['it explores', 'it investigates', 'it studies'])} how these {random.choice(['principles', 'ideas', 'elements'])} interact." 
                })
            else:  # dialogues
                examples.append({
                    "instruction": f"Discuss: {topic}",
                    "response": f"Person A: What is {topic}?\nPerson B: It's a field that {random.choice(['examines', 'studies', 'investigates'])} {topic.split()[-1]}.\nPerson A: Why is it important?\nPerson B: Because it helps us understand {random.choice(['complex systems', 'natural phenomena', 'human behavior'])}." 
                })
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]["instruction"], self.examples[idx]["response"]

# Create datasets for each curriculum stage
print("Creating datasets...")
datasets = {}
for stage_config in CURRICULUM:
    stage = stage_config["name"]
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
            
            # 2. Semantic Loss - simplified for CPU
            # Using a placeholder since we don't have hidden states
            sem_loss += torch.tensor(0.5, device=device)  # Fixed penalty
            
            # 3. Diversity Loss - simplified for CPU
            div_loss += torch.tensor(0.2, device=device)  # Fixed penalty
            
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

# Validation function
def validate(val_loader, curriculum_stage):
    model.eval()
    val_metrics = {
        "efficiency": 0.0,
        "reference": 0.0,
        "count": 0
    }
    
    with torch.no_grad():
        for instructions, responses in val_loader:
            # Tokenize inputs
            max_length = next(s for s in CURRICULUM if s["name"] == curriculum_stage)["max_length"]
            inputs = tokenizer(
                instructions, 
                padding=True, 
                truncation=True, 
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # Generate responses
            gen_outputs = model.generate(
                inputs.input_ids,
                max_length=max_length + target_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            gen_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            
            # Calculate metrics
            batch_size = len(instructions)
            val_metrics["count"] += batch_size
            
            for i in range(batch_size):
                gen_response = gen_texts[i][len(instructions[i]):].strip()
                output_length = len(tokenizer.encode(gen_response))
                
                # Efficiency
                length_diff = output_length - target_length
                val_metrics["efficiency"] += F.softplus(torch.tensor(length_diff)).item()
                
                # Reference-based metrics
                try:
                    bleu = calculate_bleu(responses[i], gen_response)
                    meteor = calculate_meteor(responses[i], gen_response)
                    val_metrics["reference"] += 1.0 - (0.5 * bleu + 0.5 * meteor)
                except:
                    val_metrics["reference"] += 1.0  # Max penalty if error
    
    # Average metrics
    for key in val_metrics:
        if key != "count":
            val_metrics[key] /= val_metrics["count"]
    
    model.train()
    return val_metrics

# Main training loop with curriculum
total_epochs = sum(stage["epochs"] for stage in CURRICULUM)
current_stage_idx = 0
global_step = 0
best_val_loss = float('inf')

# Create output directory
os.makedirs("checkpoints", exist_ok=True)

for stage in CURRICULUM:
    stage_name = stage["name"]
    stage_epochs = stage["epochs"]
    print(f"\n=== Starting curriculum stage: {stage_name} ({stage_epochs} epochs) ===")
    
    # Create dataset and split
    full_dataset = datasets[stage_name]
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders with small batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # Very small batch for CPU testing
        shuffle=True,
        collate_fn=lambda batch: list(zip(*batch))  # Unzip into instructions/responses
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2,
        collate_fn=lambda batch: list(zip(*batch))
    
    for epoch in range(stage_epochs):
        print(f"\nEpoch {epoch+1}/{stage_epochs} - Stage: {stage_name}")
        
        # Train for one epoch
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # Adjust lambda_reward over time
            decayed_lambda = lambda_reward * (1 - global_step / (total_epochs * len(train_loader)))
            
            metrics = train_step(batch, stage_name, decayed_lambda)
            global_step += 1
            
            if batch_idx % 5 == 0:  # More frequent logging
                print(f"Batch {batch_idx}: CE Loss: {metrics['loss_ce']:.4f} | "
                      f"Reward: {metrics['reward_loss']:.4f} | "
                      f"Eff: {metrics['efficiency']:.2f} | "
                      f"Ref: {metrics['reference']:.4f}")
        
        # Validate and save checkpoint
        val_metrics = validate(val_loader, stage_name)
        val_loss = val_metrics["efficiency"] + val_metrics["reference"]
        
        print(f"\nValidation - Stage: {stage_name}")
        print(f"Efficiency: {val_metrics['efficiency']:.2f} | "
              f"Reference: {val_metrics['reference']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"checkpoints/best_model_{stage_name}.pt")
            print(f"Saved best model for stage {stage_name}")

# Final evaluation
print("\n=== Final Evaluation ===")
test_concepts = [
    "photosynthesis", 
    "neural networks", 
    "quantum entanglement"
]

for concept in test_concepts:
    prompt = f"Explain {concept} in simple terms:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            temperature=0.7
        )
    
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response = output_text[len(prompt):].strip()
    response_length = len(tokenizer.encode(response))
    
    print(f"\nConcept: {concept}")
    print(f"Length: {response_length} tokens")
    print(f"Response: {response}")
    print("-" * 80)
