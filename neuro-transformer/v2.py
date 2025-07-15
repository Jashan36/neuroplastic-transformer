import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import nltk
from torch.cuda.amp import autocast, GradScaler
import random
import os
import json
import csv
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import pipeline
from dataclasses import dataclass, field
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Download required resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter configuration using dataclass
@dataclass
class TrainingConfig:
    target_length: int = 50
    plasticity_lr: float = 1e-6
    gamma: float = 0.95
    beta: float = 1.0
    lambda_decay: float = 0.01
    lambda_reward: float = 0.2
    alpha: float = 5.0   # Efficiency
    beta_sem: float = 0.7  # Semantic
    gamma_div: float = 0.3  # Diversity
    delta_ref: float = 0.4  # Reference-based
    epsilon_fact: float = 0.5  # Factual consistency
    curriculum: list = field(default_factory=lambda: [
        {"name": "definitions", "max_length": 30, "epochs": 1, "domains": ["science", "humanities", "code"]},
        {"name": "explanations", "max_length": 80, "epochs": 2, "domains": ["science", "humanities", "code", "finance", "legal"]},
        {"name": "dialogues", "max_length": 150, "epochs": 1, "domains": ["science", "humanities", "code", "finance", "legal", "medical", "conversation"]}
    ])

config = TrainingConfig()

# Initialize scorers once (outside loops)
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
NLI_PIPELINE = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

# Initialize model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
tokenizer.pad_token = tokenizer.eos_token

# Plasticity Wrapper with improved naming
class NeuroPlasticWrapper(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.plasticity_traces = {}
        self.trace_history = {}
        
        for name, param in self.model.named_parameters():
            if "weight" in name and "ln" not in name and "embed" not in name:
                self.plasticity_traces[name] = torch.zeros_like(param.data)
                self.trace_history[name] = []
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def update_plasticity_traces(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.plasticity_traces and param.grad is not None:
                    grad = param.grad.detach()
                    grad_norm = torch.norm(grad)
                    normalized_grad = grad / (grad_norm + 1e-8)
                    self.plasticity_traces[name] = (
                        self.config.gamma * self.plasticity_traces[name] + 
                        torch.abs(normalized_grad)
                    )
                    trace_norm = torch.norm(self.plasticity_traces[name]).item()
                    self.trace_history[name].append(trace_norm)
    
    def plastic_update(self, plasticity_gradients):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.plasticity_traces:
                    modulatory_signal = plasticity_gradients.get(name, 0)
                    trace = torch.clamp(self.plasticity_traces[name], min=-1, max=1)
                    plastic_update = self.config.plasticity_lr * (
                        self.config.beta * modulatory_signal * trace -
                        self.config.lambda_decay * param.data
                    )
                    param.data += plastic_update
    
    def visualize_traces(self, step, output_dir="trace_visualizations"):
        """Visualize plasticity trace evolution"""
        os.makedirs(output_dir, exist_ok=True)
        for name, history in self.trace_history.items():
            if len(history) > 10:
                plt.figure(figsize=(10, 4))
                plt.plot(history)
                plt.title(f"Plasticity Trace Norm: {name}")
                plt.xlabel("Training Steps")
                plt.ylabel("Trace Norm")
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"trace_{name}_step{step}.png"))
                plt.close()

# Initialize wrapper and optimizer
wrapper = NeuroPlasticWrapper(model, config).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Initialize GradScaler only if CUDA is available
if device.type == 'cuda':
    scaler = GradScaler()
else:
    scaler = None
    print("CUDA not available - disabling mixed precision training")

# Enhanced reward calculation functions
def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores between reference and candidate"""
    scores = ROUGE_SCORER.score(reference, candidate)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

def calculate_factual_consistency(prompt, response, method='nli'):
    """Calculate factual consistency using NLI or fallback method"""
    if method == 'nli' and NLI_PIPELINE:
        try:
            # Use NLI to check if response is entailed by prompt
            result = NLI_PIPELINE(
                sequence=response,
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis_template="This text contains: {}",
                multi_label=False
            )
            entailment_score = result['scores'][result['labels'].index('entailment')]
            return entailment_score
        except Exception as e:
            print(f"NLI pipeline error: {e}")
            return calculate_factual_consistency_simple(prompt, response)
    else:
        return calculate_factual_consistency_simple(prompt, response)

def calculate_factual_consistency_simple(prompt, response):
    """Fallback method using entity matching"""
    prompt_entities = set()
    response_entities = set()
    prompt_pos = nltk.pos_tag(nltk.word_tokenize(prompt))
    response_pos = nltk.pos_tag(nltk.word_tokenize(response))
    
    # Extract nouns and proper nouns
    for word, tag in prompt_pos:
        if tag.startswith('NN'):
            prompt_entities.add(word.lower())
    
    for word, tag in response_pos:
        if tag.startswith('NN'):
            response_entities.add(word.lower())
    
    new_entities = response_entities - prompt_entities
    return 1.0 if len(new_entities) == 0 else 0.0

# Enhanced dataset with more domains and context handling
class CurriculumDataset(Dataset):
    def __init__(self, stage="definitions", domains=None):
        self.examples = []
        self.domains = domains or []
        
        # Load from local CSV file
        try:
            with open('dolly_data.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for item in reader:
                    if item.get('domain', '') in self.domains:
                        if stage == "definitions" and len(item['response']) < 100:
                            self.examples.append({
                                "instruction": f"Define: {item['instruction']}",
                                "response": item['response'],
                                "domain": item.get('domain', 'unknown')
                            })
                        elif stage == "explanations" and len(item['response']) >= 100:
                            self.examples.append({
                                "instruction": item['instruction'],
                                "response": item['response'],
                                "domain": item.get('domain', 'unknown')
                            })
                        elif stage == "dialogues":
                            # Add context for dialogue training
                            context = item.get('context', '')
                            self.examples.append({
                                "instruction": f"{context}Explain in a dialogue: {item['instruction']}",
                                "response": item['response'],
                                "domain": item.get('domain', 'unknown')
                            })
            print(f"Loaded {len(self.examples)} examples from local file for stage: {stage}, domains: {domains}")
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            # Fallback to synthetic data with more domains
            self.examples = self.generate_synthetic_examples(stage, 1000, domains)
            print(f"Generated {len(self.examples)} synthetic examples for stage: {stage}, domains: {domains}")
    
    def generate_synthetic_examples(self, stage, count, domains):
        """Generate synthetic training examples for given domains"""
        examples = []
        domain_topics = {
            "science": ["machine learning", "quantum physics", "biology"],
            "humanities": ["renaissance art", "ancient history", "literary analysis"],
            "code": ["software engineering", "algorithms", "data structures"],
            "finance": ["stock market", "investment", "banking"],
            "legal": ["contract law", "intellectual property", "criminal justice"],
            "medical": ["anatomy", "pharmacology", "medical ethics"],
            "conversation": ["casual talk", "daily life", "hobbies"]
        }
        
        for _ in range(count):
            # Choose a domain and then a topic within that domain
            domain = random.choice(domains) if domains else "science"
            topic = random.choice(domain_topics.get(domain, ["general topic"]))
            if stage == "definitions":
                examples.append({
                    "instruction": f"Define: {topic}",
                    "response": f"{topic} is the study of...",
                    "domain": domain
                })
            elif stage == "explanations":
                examples.append({
                    "instruction": f"Explain: {topic}",
                    "response": f"{topic} involves several key concepts. First...",
                    "domain": domain
                })
            else:  # dialogues
                # Add context for multi-turn conversations
                context = random.choice([
                    "Based on our previous discussion about related topics, ",
                    "Continuing from our last conversation, ",
                    ""
                ])
                examples.append({
                    "instruction": f"{context}Discuss: {topic}",
                    "response": f"Person A: What is {topic}?\nPerson B: It's a field that...",
                    "domain": domain
                })
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]["instruction"], self.examples[idx]["response"], self.examples[idx]["domain"]

# Create datasets for each curriculum stage
datasets = {}
for stage in set(s["name"] for s in config.curriculum):
    domains = next(s["domains"] for s in config.curriculum if s["name"] == stage)
    datasets[stage] = CurriculumDataset(stage, domains)

# Named collate function for clarity
def collate_batch(batch):
    """Collates batch of (instruction, response, domain) tuples"""
    instructions, responses, domains = zip(*batch)
    return list(instructions), list(responses), list(domains)

# Enhanced training function with context handling
def train_step(batch, curriculum_stage, global_step):
    instructions, responses, domains = batch
    
    # Tokenize with curriculum-based max length
    max_length = next(s for s in config.curriculum if s["name"] == curriculum_stage)["max_length"]
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
    
    # Mixed precision context
    use_autocast = device.type == 'cuda' and scaler is not None
    with autocast(enabled=use_autocast):
        # --- STANDARD LANGUAGE MODELING ---
        outputs = model(**inputs, labels=response_inputs.input_ids)
        loss_ce = outputs.loss
        
        # --- GENERATE RESPONSE FOR REWARD CALCULATION ---
        gen_outputs = model.generate(
            inputs.input_ids,
            max_length=max_length + config.target_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            repetition_penalty=1.2
        )
        gen_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
        
        # Differentiable forward pass on generated sequences
        model_outputs = model(gen_outputs, output_hidden_states=True)
        hidden_states = model_outputs.hidden_states[-1] if model_outputs.hidden_states is not None else None
        
        # Initialize reward components
        eff_loss = torch.tensor(0.0, device=device)
        sem_loss = torch.tensor(0.0, device=device)
        div_loss = torch.tensor(0.0, device=device)
        ref_loss = torch.tensor(0.0, device=device)
        fact_loss = torch.tensor(0.0, device=device)
        
        # Calculate rewards per item in batch
        for i in range(len(gen_texts)):
            prompt_length = inputs.input_ids[i].shape[0]
            gen_response = gen_texts[i][len(instructions[i]):].strip()
            output_length = len(tokenizer.encode(gen_response))
            
            # 1. Efficiency Loss
            length_diff = output_length - config.target_length
            item_eff = F.softplus(torch.tensor(length_diff, device=device))
            eff_loss += torch.clamp(item_eff, max=20.0)
            
            # 2. Semantic Loss
            if hidden_states is not None:
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
            
            # 4. Reference-based Loss (ROUGE + METEOR)
            try:
                rouge1, rougeL = calculate_rouge(responses[i], gen_response)
                meteor_score_val = meteor_score([responses[i]], gen_response)
                ref_loss += 1.0 - (0.4 * rouge1 + 0.4 * rougeL + 0.2 * meteor_score_val)
            except Exception as e:
                print(f"Error calculating reference metrics: {e}")
                ref_loss += torch.tensor(1.0, device=device)  # Max penalty if error
            
            # 5. Factual Consistency Loss
            fact_score = calculate_factual_consistency(instructions[i], gen_response)
            fact_loss += 1.0 - fact_score
        
        # Average losses across batch
        batch_size = len(instructions)
        eff_loss /= batch_size
        sem_loss /= batch_size
        div_loss /= batch_size
        ref_loss /= batch_size
        fact_loss /= batch_size
        
        # Combined reward loss
        reward_loss = (
            config.alpha * eff_loss + 
            config.beta_sem * sem_loss + 
            config.gamma_div * div_loss +
            config.delta_ref * ref_loss +
            config.epsilon_fact * fact_loss
        )
        
        # Combined total loss
        total_loss = loss_ce + config.lambda_reward * reward_loss
    
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
    
    # Update plasticity traces
    wrapper.update_plasticity_traces()
    
    # Prepare plasticity update signals
    plasticity_gradients = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in wrapper.plasticity_traces and param.grad is not None:
                plasticity_gradients[name] = -param.grad.detach()
    
    # Apply neuroplastic update
    wrapper.plastic_update(plasticity_gradients)
    
    # Visualize traces periodically
    if global_step % 100 == 0:
        wrapper.visualize_traces(global_step)
    
    return {
        "loss_ce": loss_ce.item(),
        "reward_loss": reward_loss.item(),
        "efficiency": eff_loss.item(),
        "semantic": sem_loss.item(),
        "diversity": -div_loss.item(),
        "reference": ref_loss.item(),
        "factual": fact_loss.item()
    }

# Enhanced validation function with more metrics
def validate(val_loader, curriculum_stage):
    model.eval()
    val_metrics = {
        "loss_ce": 0.0,
        "efficiency": 0.0,
        "semantic": 0.0,
        "diversity": 0.0,
        "reference": 0.0,
        "factual": 0.0,
        "count": 0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            instructions, responses, domains = collate_batch(batch)
            
            # Tokenize
            max_length = next(s for s in config.curriculum if s["name"] == curriculum_stage)["max_length"]
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
                max_length=max_length + config.target_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50
            )
            gen_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            
            # Calculate metrics
            batch_size = len(instructions)
            val_metrics["count"] += batch_size
            
            # CE Loss
            response_inputs = tokenizer(
                responses, 
                padding=True, 
                truncation=True, 
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs, labels=response_inputs.input_ids)
            val_metrics["loss_ce"] += outputs.loss.item() * batch_size
            
            for i in range(batch_size):
                gen_response = gen_texts[i][len(instructions[i]):].strip()
                output_length = len(tokenizer.encode(gen_response))
                
                # Efficiency
                length_diff = output_length - config.target_length
                val_metrics["efficiency"] += F.softplus(torch.tensor(length_diff)).item()
                
                # Reference-based metrics
                try:
                    rouge1, rougeL = calculate_rouge(responses[i], gen_response)
                    meteor = meteor_score([responses[i]], gen_response)
                    ref_score = 0.4 * rouge1 + 0.4 * rougeL + 0.2 * meteor
                    val_metrics["reference"] += 1.0 - ref_score
                except Exception as e:
                    print(f"Validation metric error: {e}")
                    val_metrics["reference"] += 1.0
                
                # Factual consistency
                fact_score = calculate_factual_consistency(instructions[i], gen_response)
                val_metrics["factual"] += 1.0 - fact_score
    
    # Average metrics
    for key in val_metrics:
        if key != "count":
            val_metrics[key] /= val_metrics["count"]
    
    model.train()
    return val_metrics

# Inference interface class
class InferenceInterface:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
    def generate_response(self, prompt, max_length=150, temperature=0.7, top_k=50):
        """Generate response to a given prompt"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                temperature=temperature,
                top_k=top_k
            )
        
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = output_text[len(prompt):].strip()
        return response
    
    def batch_process(self, input_file, output_file):
        """Process a file with multiple prompts"""
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                prompt = line.strip()
                if prompt:
                    response = self.generate_response(prompt)
                    f_out.write(f"Prompt: {prompt}\nResponse: {response}\n\n")
    
    def interactive_chat(self):
        """Interactive chat interface"""
        print("Starting interactive chat. Type 'quit' to exit.")
        context = ""
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
                
            # Maintain context for multi-turn conversations
            full_prompt = f"{context}User: {user_input}\nAssistant:"
            response = self.generate_response(full_prompt)
            print(f"Assistant: {response}")
            
            # Update context (keep last 3 exchanges)
            context += f"User: {user_input}\nAssistant: {response}\n"
            context_lines = context.split('\n')
            if len(context_lines) > 6:  # Keep last 3 exchanges
                context = '\n'.join(context_lines[-6:])

# Enhanced OOD evaluation
def ood_evaluation(model, tokenizer):
    print("\n=== Quantitative Out-of-Distribution Evaluation ===")
    ood_prompts = [
        ("Write a joke about artificial intelligence:", "jokes"),
        ("Compose a haiku about the ocean:", "poetry"),
        ("Explain quantum computing using only metaphors:", "creative"),
        ("Describe the API specifications for a RESTful service:", "tech_specs"),
        ("Write a dramatic monologue about climate change:", "drama")
    ]
    
    results = []
    model.eval()
    for prompt, category in ood_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 100,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                temperature=0.8
            )
        
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = output_text[len(prompt):].strip()
        
        # Calculate real metrics
        rouge1, rougeL = calculate_rouge(prompt, response)
        meteor_val = meteor_score([prompt], response)
        fact_score = calculate_factual_consistency(prompt, response)
        response_length = len(tokenizer.encode(response))
        
        print(f"\nCategory: {category}")
        print(f"ROUGE-L: {rougeL:.4f} | Factual: {fact_score:.4f} | Length: {response_length}")
        print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        results.append({
            "category": category,
            "rougeL": rougeL,
            "factual": fact_score,
            "length": response_length,
            "response": response
        })
    return results

# Main training loop with enhanced logging
def main_training_loop():
    total_epochs = sum(stage["epochs"] for stage in config.curriculum)
    global_step = 0
    best_val_loss = float('inf')
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Create log file with header
    with open(log_file, 'w') as f:
        f.write("timestamp,stage,epoch,batch,loss_ce,reward_loss,efficiency,semantic,diversity,reference,factual\n")
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    for stage in config.curriculum:
        stage_name = stage["name"]
        stage_epochs = stage["epochs"]
        print(f"\nStarting curriculum stage: {stage_name} ({stage_epochs} epochs)")
        
        # Create dataset and split
        full_dataset = datasets[stage_name]
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=8, 
            shuffle=True,
            collate_fn=collate_batch
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=8,
            collate_fn=collate_batch
        )

        for epoch in range(stage_epochs):
            print(f"\nEpoch {epoch+1}/{stage_epochs} - Stage: {stage_name}")
            
            # Train for one epoch
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # Adjust lambda_reward over time
                decayed_lambda = config.lambda_reward * (1 - global_step / (total_epochs * len(train_loader)))
                
                metrics = train_step(batch, stage_name, global_step)
                global_step += 1
                
                # Log metrics
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(log_file, 'a') as f:
                    f.write(f"{timestamp},{stage_name},{epoch+1},{batch_idx},"
                            f"{metrics['loss_ce']:.4f},{metrics['reward_loss']:.4f},"
                            f"{metrics['efficiency']:.4f},{metrics['semantic']:.4f},"
                            f"{metrics['diversity']:.4f},{metrics['reference']:.4f},"
                            f"{metrics['factual']:.4f}\n")
                
                if batch_idx % 20 == 0:
                    print(f"Batch {batch_idx}: CE Loss: {metrics['loss_ce']:.4f} | "
                          f"Reward: {metrics['reward_loss']:.4f} | "
                          f"Eff: {metrics['efficiency']:.2f} | "
                          f"Fact: {metrics['factual']:.4f}")
            
            # Validate and save checkpoint
            val_metrics = validate(val_loader, stage_name)
            val_loss = val_metrics["loss_ce"] + decayed_lambda * (
                config.alpha * val_metrics["efficiency"] +
                config.beta_sem * val_metrics["semantic"] +
                config.gamma_div * (1 - val_metrics["diversity"]) +
                config.delta_ref * val_metrics["reference"] +
                config.epsilon_fact * val_metrics["factual"]
            )
            
            print(f"\nValidation - Stage: {stage_name}")
            print(f"CE Loss: {val_metrics['loss_ce']:.4f} | "
                  f"Efficiency: {val_metrics['efficiency']:.2f} | "
                  f"Factual: {val_metrics['factual']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"checkpoints/best_model_{stage_name}.pt")
                print(f"Saved best model for stage {stage_name}")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'global_step': global_step,
                'config': config
            }, f"checkpoints/checkpoint_{stage_name}_epoch{epoch}.pt")

    return model

if __name__ == "__main__":
    # Run training
    trained_model = main_training_loop()
    
    # Load best model for final evaluation
    try:
        trained_model.load_state_dict(torch.load("checkpoints/best_model_dialogues.pt"))
    except:
        print("Could not load best model, using last trained model")
    
    # Final evaluation on diverse concepts
    print("\n=== Final Evaluation ===")
    test_concepts = [
        ("photosynthesis", "science"), 
        ("Shakespeare's influence", "humanities"), 
        ("Python decorators", "code"),
        ("quantum entanglement", "science"),
        ("mortgage-backed securities", "finance"),
        ("HIPAA compliance", "legal"),
        ("myocardial infarction", "medical")
    ]
    
    inference = InferenceInterface(trained_model, tokenizer)
    
    for concept, domain in test_concepts:
        prompt = f"Explain {concept} in simple terms:"
        response = inference.generate_response(prompt)
        response_length = len(tokenizer.encode(response))
        fact_score = calculate_factual_consistency(prompt, response)
        
        print(f"\nConcept: {concept} ({domain})")
        print(f"Length: {response_length} tokens | Factual: {fact_score:.2f}")
        print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
    
    # Out-of-distribution evaluation
    ood_results = ood_evaluation(trained_model, tokenizer)
    
    # Save OOD results
    with open("ood_evaluation.json", "w") as f:
        json.dump(ood_results, f, indent=2)
    
    # Initialize inference interface
    print("\n=== Inference Interface ===")
    inference_engine = InferenceInterface(trained_model, tokenizer)
    
    # Example usage:
    # inference_engine.interactive_chat()
    # inference_engine.batch_process("input_prompts.txt", "output_responses.txt")
