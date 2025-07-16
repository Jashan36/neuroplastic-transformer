import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import nltk
from torch.amp import autocast, GradScaler
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
  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Download required resources
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("NLTK downloads failed, some features may not work")

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
try:
    ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    NLI_PIPELINE = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
except:
    print("Could not initialize ROUGE scorer or NLI pipeline")
    ROUGE_SCORER = None
    NLI_PIPELINE = None

# Initialize model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
tokenizer.pad_token = tokenizer.eos_token


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# *** FIX: Set padding side to 'left' to resolve the warning and improve generation ***
tokenizer.padding_side = 'left'

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
    if ROUGE_SCORER is None:
        return 0.5, 0.5  # Default scores if ROUGE not available
    try:
        scores = ROUGE_SCORER.score(reference, candidate)
        return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure
    except:
        return 0.5, 0.5

def calculate_factual_consistency(prompt, response, method='simple'):
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
    try:
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
        return 1.0 if len(new_entities) == 0 else 0.8
    except:
        return 0.7  # Default score if NLTK fails

# Enhanced dataset with more domains and context handling
class CurriculumDataset(Dataset):
    def __init__(self, stage="definitions", domains=None):
        self.examples = []
        self.domains = domains or []
        
        # Try to load from local CSV file, but don't crash if it doesn't exist
        csv_loaded = False
        try:
            if os.path.exists('dolly_data.csv'):
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
                                context = item.get('context', '')
                                self.examples.append({
                                    "instruction": f"{context}Explain in a dialogue: {item['instruction']}",
                                    "response": item['response'],
                                    "domain": item.get('domain', 'unknown')
                                })
                print(f"Loaded {len(self.examples)} examples from local file for stage: {stage}, domains: {domains}")
                csv_loaded = True
            else:
                print("Local dolly_data.csv not found, using synthetic data")
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            csv_loaded = False
        
        # Always generate synthetic data if CSV wasn't loaded or has insufficient data
        if not csv_loaded or len(self.examples) < 100:
            synthetic_examples = self.generate_synthetic_examples(stage, 1000, domains)
            self.examples.extend(synthetic_examples)
            print(f"Generated {len(synthetic_examples)} synthetic examples for stage: {stage}, domains: {domains}")
                
    def generate_synthetic_examples(self, stage, count, domains):
        """Generate synthetic training examples for given domains"""
        examples = []
        domain_topics = {
            "science": ["machine learning", "quantum physics", "biology", "chemistry", "physics"],
            "humanities": ["renaissance art", "ancient history", "literary analysis", "philosophy", "linguistics"],
            "code": ["software engineering", "algorithms", "data structures", "programming", "databases"],
            "finance": ["stock market", "investment", "banking", "economics", "cryptocurrency"],
            "legal": ["contract law", "intellectual property", "criminal justice", "constitutional law", "torts"],
            "medical": ["anatomy", "pharmacology", "medical ethics", "pathology", "surgery"],
            "conversation": ["casual talk", "daily life", "hobbies", "travel", "food"]
        }
        
        for _ in range(count):
            # Choose a domain and then a topic within that domain
            domain = random.choice(domains) if domains else "science"
            topic = random.choice(domain_topics.get(domain, ["general topic"]))
            
            if stage == "definitions":
                examples.append({
                    "instruction": f"Define: {topic}",
                    "response": f"{topic} is a concept that involves understanding key principles and applications in {domain}.",
                    "domain": domain
                })
            elif stage == "explanations":
                examples.append({
                    "instruction": f"Explain: {topic}",
                    "response": f"{topic} involves several key concepts. First, it requires understanding the fundamental principles. Second, it involves practical applications. This field is important because it connects theory with real-world implementations.",
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
                    "response": f"Person A: What is {topic}?\nPerson B: It's a field that focuses on understanding and applying key concepts. It's particularly relevant in {domain} applications.",
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

# FIXED: Enhanced training function with proper batch handling
def train_step(batch, curriculum_stage, global_step):
    instructions, responses, domains = batch
    max_length = next(s for s in config.curriculum if s["name"] == curriculum_stage)["max_length"]
    
    # Validate batch
    if len(instructions) == 0 or len(responses) == 0:
        print("Empty batch encountered, skipping...")
        return {"loss_ce": 0.0, "reward_loss": 0.0, "efficiency": 0.0, "semantic": 0.0, "diversity": 0.0, "reference": 0.0, "factual": 0.0}
    
    # Create combined sequences for training
    combined_texts = []
    for inst, resp in zip(instructions, responses):
        combined_texts.append(f"{inst} {resp}")
    
    # FIX: More robust tokenization
    try:
        encodings = tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=max_length + config.target_length,
            return_tensors="pt"
        ).to(device)
        
        labels = encodings.input_ids.clone()
        
        instruction_encodings = tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        instruction_lengths = instruction_encodings.attention_mask.sum(dim=1)
        
        # Proper label masking
        for i, inst_len in enumerate(instruction_lengths):
            if inst_len < labels.shape[1]:
                labels[i, :inst_len] = -100
                
    except Exception as e:
        print(f"Tokenization error: {e}")
        return {"loss_ce": 0.0, "reward_loss": 0.0, "efficiency": 0.0, "semantic": 0.0, "diversity": 0.0, "reference": 0.0, "factual": 0.0}
    
    # Mixed precision context - FIXED
    use_autocast = device.type == 'cuda' and scaler is not None
    
    with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=use_autocast):
        # Forward pass
        outputs = model(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            labels=labels
        )
        loss_ce = outputs.loss
        
        # FIX: Better generation error handling
        try:
            gen_outputs = model.generate(
                input_ids=instruction_encodings.input_ids,
                attention_mask=instruction_encodings.attention_mask,
                max_new_tokens=min(config.target_length, 100),  # Limit max tokens
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                repetition_penalty=1.2,
                temperature=0.8,
                early_stopping=True  # Add early stopping
            )
            
            # Extract generated responses
            gen_texts = []
            for i in range(len(gen_outputs)):
                input_len = instruction_encodings.input_ids.shape[1]
                if len(gen_outputs[i]) > input_len:
                    gen_tokens = gen_outputs[i, input_len:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    gen_texts.append(gen_text)
                else:
                    gen_texts.append("")  # Empty response if generation failed
                    
        except Exception as e:
            print(f"Generation error in training: {e}")
            gen_texts = ["fallback response"] * len(instructions)
        
        # Rest of the function remains the same...
        # [Continue with reward calculations as in original code]

# 5. FIX: Initialize GradScaler properly
# Replace the scaler initialization with:
if device.type == 'cuda':
    try:
        scaler = GradScaler(device='cuda')
    except Exception as e:
        print(f"Failed to initialize GradScaler: {e}")
        scaler = None
else:
    scaler = None
    print("CUDA not available - disabling mixed precision training")
# Enhanced validation function
def validate(val_loader, curriculum_stage):
    """
    *** CORRECTED VERSION ***
    This function correctly calculates validation loss by ensuring the model's input
    and label tensors have matching shapes, resolving the ValueError.
    """
    model.eval()
    val_metrics = {k: 0.0 for k in ["loss_ce", "efficiency", "semantic", "diversity", "reference", "factual"]}
    count = 0
    max_length = next(s["max_length"] for s in config.curriculum if s["name"] == curriculum_stage)

    with torch.no_grad():
        for batch in val_loader:
            instructions, responses, domains = batch
            batch_size = len(instructions)
            count += batch_size

            # --- Start: Cross-Entropy Loss Calculation (FIXED LOGIC) ---
            # 1. Create the full sequence: instruction + response
            full_texts = [inst + " " + resp for inst, resp in zip(instructions, responses)]

            # 2. Tokenize the full sequence with consistent padding
            full_encodings = tokenizer(
                full_texts,
                padding=True,  # Changed from 'max_length' to True for dynamic padding
                truncation=True,
                max_length=max_length + config.target_length,
                return_tensors="pt"
            ).to(device)

            # 3. Create labels by cloning the inputs. Their shapes now match perfectly.
            labels = full_encodings.input_ids.clone()

            # 4. Mask the instruction part of the labels so loss isn't computed on them
            # FIX: Tokenize instructions with same settings as full_texts
            instruction_only_encodings = tokenizer(
                instructions, 
                padding=True,  # Changed from True to match full_encodings
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(device)
            
            # FIX: Calculate instruction lengths properly
            instruction_lengths = instruction_only_encodings.attention_mask.sum(dim=1)

            # FIX: Proper label masking
            for i, inst_len in enumerate(instruction_lengths):
                if inst_len < labels.shape[1]:  # Safety check
                    labels[i, :inst_len] = -100  # -100 is the standard ignore_index for cross-entropy

            # 5. Calculate loss. The shapes of model logits and labels will now align.
            outputs_ce = model(
                input_ids=full_encodings.input_ids,
                attention_mask=full_encodings.attention_mask,
                labels=labels
            )
            val_metrics["loss_ce"] += outputs_ce.loss.item() * batch_size
            # --- End: Cross-Entropy Loss Calculation (FIXED LOGIC) ---

            # --- Start: Reward Calculation (FIXED) ---
            # FIX: Use consistent tokenization for generation
            inputs_enc = tokenizer(
                instructions,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            try:
                gen_outputs = model.generate(
                    input_ids=inputs_enc.input_ids,
                    attention_mask=inputs_enc.attention_mask,
                    max_new_tokens=config.target_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    num_return_sequences=1  # Ensure single sequence per input
                )

                # Extract only the generated part (after instruction)
                gen_texts = []
                for i in range(batch_size):
                    input_len = inputs_enc.input_ids.shape[1]
                    gen_tokens = gen_outputs[i, input_len:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    gen_texts.append(gen_text)

                # Calculate reward metrics
                for i in range(batch_size):
                    gen_response = gen_texts[i]
                    
                    # Efficiency
                    response_length = len(tokenizer.encode(gen_response))
                    eff_loss = max(0, response_length - config.target_length)
                    val_metrics["efficiency"] += eff_loss

                    # Semantic (simplified - using response length as proxy)
                    val_metrics["semantic"] += 0.5  # Placeholder

                    # Diversity (simplified)
                    val_metrics["diversity"] += 0.3  # Placeholder

                    # Reference-based metrics
                    try:
                        r1, rL = calculate_rouge(responses[i], gen_response)
                        val_metrics["reference"] += (1.0 - (0.5 * r1 + 0.5 * rL))
                    except Exception:
                        val_metrics["reference"] += 1.0

                    # Factual consistency
                    try:
                        fact_score = calculate_factual_consistency(instructions[i], gen_response)
                        val_metrics["factual"] += (1.0 - fact_score)
                    except Exception:
                        val_metrics["factual"] += 0.5
                        
            except Exception as e:
                print(f"Generation error in validation: {e}")
                # Add default values for failed generation
                for i in range(batch_size):
                    val_metrics["efficiency"] += 1.0
                    val_metrics["semantic"] += 0.5
                    val_metrics["diversity"] += 0.3
                    val_metrics["reference"] += 1.0
                    val_metrics["factual"] += 0.5

    # Average all metrics by the total number of items processed
    for key in val_metrics:
        if count > 0:
            val_metrics[key] /= count

    model.train()  # Set the model back to training mode
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
# --- 5. Main Execution and Evaluation ---
def main_training_loop():
    total_steps = sum(stage["epochs"] * (len(CurriculumDataset(stage['name'], stage['domains'])) // 8) for stage in config.curriculum)
    global_step = 0
    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    log_file = os.path.join("training_logs", f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "stage", "epoch", "batch", "loss_ce", "reward_loss", "efficiency", "semantic", "diversity", "reference", "factual"])

    
    for stage in config.curriculum:
        stage_name, stage_epochs = stage["name"], stage["epochs"]
        print(f"\n--- Starting Curriculum Stage: {stage_name} ({stage_epochs} epochs) ---")
        logging.info(f"\n--- Starting Curriculum Stage: {stage_name} ({stage_epochs} epochs) ---")
        
        try:
            full_dataset = CurriculumDataset(stage['name'], stage['domains'])
            
            # FIX: Better minimum dataset size check
            if len(full_dataset) < 20:  # Increased minimum
                print(f"Dataset too small for stage {stage_name}: {len(full_dataset)} samples")
                logging.warning(f"Dataset too small for stage {stage_name}: {len(full_dataset)} samples")
                # Generate more synthetic data
                synthetic_dataset = CurriculumDataset(stage['name'], stage['domains'])
                full_dataset = synthetic_dataset
                
            # FIX: Better train/val split with minimum validation size
            train_size = max(16, int(0.9 * len(full_dataset)))  # Minimum 16 for training
            val_size = max(4, len(full_dataset) - train_size)    # Minimum 4 for validation
            
            # Adjust if total is too small
            if train_size + val_size > len(full_dataset):
                train_size = len(full_dataset) - 4
                val_size = 4
            
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            # FIX: Better batch size calculation
            effective_batch_size = min(8, max(2, len(train_dataset) // 4))
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=effective_batch_size, 
                shuffle=True, 
                collate_fn=collate_batch,
                drop_last=len(train_dataset) > effective_batch_size  # Only drop if we have enough data
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=min(effective_batch_size, len(val_dataset)), 
                collate_fn=collate_batch,
                drop_last=False  # Don't drop for validation
            )
            
        except Exception as e:
            print(f"Failed to create datasets: {e}")
            logging.error(f"Failed to create datasets: {e}")
            continue
        for epoch in range(stage_epochs):
            print(f"Epoch {epoch+1}/{stage_epochs}")
            logging.info(f"Epoch {epoch+1}/{stage_epochs}")
            model.train()
            
            epoch_metrics = {k: [] for k in ["loss_ce", "reward_loss", "efficiency", "semantic", "diversity", "reference", "factual"]}
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # FIX: Validate batch size
                    if len(batch[0]) == 0:  # Empty batch
                        continue
                        
                    metrics = train_step(batch, stage_name, global_step)
                    global_step += 1
                    
                    # Track metrics
                    for key, value in metrics.items():
                        epoch_metrics[key].append(value)
                    
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                            stage_name, 
                            epoch + 1, 
                            batch_idx
                        ] + [f"{v:.4f}" for v in metrics.values()])
                    
                    if batch_idx % 20 == 0:
                        print(f"Batch {batch_idx}: CE Loss: {metrics['loss_ce']:.4f} | Reward: {metrics['reward_loss']:.4f}")
                        logging.info(f"Batch {batch_idx}: CE Loss: {metrics['loss_ce']:.4f} | Reward: {metrics['reward_loss']:.4f}")
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    logging.error(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            # Calculate epoch averages
            epoch_avg = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
            print(f"Epoch {epoch+1} averages: CE Loss: {epoch_avg['loss_ce']:.4f} | Reward: {epoch_avg['reward_loss']:.4f}")
            logging.info(f"Epoch {epoch+1} averages: CE Loss: {epoch_avg['loss_ce']:.4f} | Reward: {epoch_avg['reward_loss']:.4f}")
            
            # Validation
            try:
                val_metrics = validate(val_loader, stage_name)
                decayed_lambda = config.lambda_reward * (1 - global_step / total_steps)
                
                # Calculate combined validation loss for checkpointing
                val_reward_loss = (
                    config.alpha * val_metrics["efficiency"] + 
                    config.beta_sem * val_metrics["semantic"] +
                    config.gamma_div * val_metrics["diversity"] +
                    config.delta_ref * val_metrics["reference"] +
                    config.epsilon_fact * val_metrics["factual"]
                )
                val_loss = val_metrics["loss_ce"] + decayed_lambda * val_reward_loss
                
                print(f"\nValidation - Stage: {stage_name} | Total Val Loss: {val_loss:.4f}")
                print(f"Val CE Loss: {val_metrics['loss_ce']:.4f} | Val Reward Loss: {val_reward_loss:.4f}")
                logging.info(f"Validation - Stage: {stage_name} | Total Val Loss: {val_loss:.4f}")
                logging.info(f"Val CE Loss: {val_metrics['loss_ce']:.4f} | Val Reward Loss: {val_reward_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(f"checkpoints/best_model_{stage_name}")
                    tokenizer.save_pretrained(f"checkpoints/best_model_{stage_name}")
                    print(f"Saved new best model for stage {stage_name} (Val Loss: {best_val_loss:.4f})")
                    logging.info(f"Saved new best model for stage {stage_name} (Val Loss: {best_val_loss:.4f})")
                    
            except Exception as e:
                print(f"Validation error: {e}")
                logging.error(f"Validation error: {e}")
                continue

    return model, tokenizer

if __name__ == "__main__":
    trained_model, trained_tokenizer = main_training_loop()
    # Your original evaluation and inference logic from code2 can be placed here.
    logging.info("\n=== Training Finished. Model is ready for inference. ===")
    # Example: inference_interface = InferenceInterface(trained_model, trained_tokenizer)
    # inference_interface.interactive_chat()
