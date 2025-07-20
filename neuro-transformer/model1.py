import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import CausalLMOutput
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import math
import json
import os
from tqdm import tqdm
import logging
from rouge_score import rouge_scorer
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegratedTrainingConfig:
    # Model architecture
    vocab_size: int = 32768  # Will be adjusted based on tokenizer
    embed_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    ffn_dim: int = 8192
    max_seq_length: int = 2048
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Neuroplasticity parameters
    plasticity_lr: float = 1e-5
    trace_decay: float = 0.9
    momentum_factor: float = 0.95
    low_rank_dim: int = 64
    modulation_strength: float = 0.1
    plasticity_update_frequency: int = 100
    
    # Curriculum and evaluation
    rouge_weight: float = 0.3
    coherence_weight: float = 0.2
    diversity_weight: float = 0.2
    perplexity_weight: float = 0.3
    curriculum_threshold: float = 0.7
    min_loss_improvement: float = 0.01
    
    # Data and checkpointing
    save_dir: str = "neuroplastic_checkpoints"
    eval_frequency: int = 500
    save_frequency: int = 1000

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture with dropout
        x = x + self.dropout(self.attention(self.ln1(x), causal_mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class CustomLanguageModel(nn.Module):
    def __init__(self, config: IntegratedTrainingConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.ffn_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Create causal mask once
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.max_seq_length, config.max_seq_length)))
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> CausalLMOutput:
        B, T = input_ids.shape
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask for current sequence length
        causal_mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                 shift_labels.view(-1), ignore_index=-100)
        
        return CausalLMOutput(loss=loss, logits=logits)

    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                 num_return_sequences: int = 1, temperature: float = 0.7, 
                 do_sample: bool = True, pad_token_id: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Simple generation method"""
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Prepare for generation
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs.logits[:, -1, :]  # Get last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit pad token or max length
                if pad_token_id is not None and next_token.item() == pad_token_id:
                    break
                    
                # Check if sequence is too long
                if generated.shape[1] >= max_length:
                    break
        
        return generated

class NeuroPlasticityMechanism(nn.Module):
    def __init__(self, model: nn.Module, config: IntegratedTrainingConfig):
        super().__init__()
        self.model = model
        self.config = config
        
        # Initialize plasticity components
        self.plasticity_traces = {}
        self.momentum_buffers = {}
        self.low_rank_components = {}
        
        self._initialize_plasticity_components()
        
    def _initialize_plasticity_components(self):
        """Initialize plasticity traces and components for all model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Plasticity traces (same shape as parameter)
                self.plasticity_traces[name] = torch.zeros_like(param.data)
                
                # Momentum buffers
                self.momentum_buffers[name] = torch.zeros_like(param.data)
                
                # Low-rank components for large matrices
                if param.dim() >= 2 and min(param.shape) > self.config.low_rank_dim:
                    rank = min(self.config.low_rank_dim, min(param.shape))
                    self.low_rank_components[name] = {
                        'U': torch.randn(param.shape[0], rank) * 0.01,
                        'V': torch.randn(rank, param.shape[1]) * 0.01
                    }
                    
                    # Move to same device as parameter
                    if param.is_cuda:
                        self.low_rank_components[name]['U'] = self.low_rank_components[name]['U'].cuda()
                        self.low_rank_components[name]['V'] = self.low_rank_components[name]['V'].cuda()
    
    def update_plasticity_traces(self, gradients: Dict[str, torch.Tensor]):
        """Update plasticity traces based on gradients"""
        for name, grad in gradients.items():
            if name in self.plasticity_traces:
                # Exponential moving average of gradients
                self.plasticity_traces[name] = (
                    self.config.trace_decay * self.plasticity_traces[name] + 
                    (1 - self.config.trace_decay) * grad.detach()
                )
    
    def compute_plasticity_modulation(self, reward_signal: float) -> Dict[str, torch.Tensor]:
        """Compute plasticity-based parameter updates"""
        modulations = {}
        
        for name, trace in self.plasticity_traces.items():
            if name in self.momentum_buffers:
                # Update momentum
                self.momentum_buffers[name] = (
                    self.config.momentum_factor * self.momentum_buffers[name] +
                    (1 - self.config.momentum_factor) * trace
                )
                
                # Compute modulation based on reward signal
                base_modulation = reward_signal * self.config.modulation_strength * self.momentum_buffers[name]
                
                # Apply low-rank approximation if available
                if name in self.low_rank_components:
                    U, V = self.low_rank_components[name]['U'], self.low_rank_components[name]['V']
                    low_rank_update = torch.mm(U, V)
                    modulations[name] = base_modulation + 0.1 * low_rank_update
                else:
                    modulations[name] = base_modulation
        
        return modulations

class AdvancedRewardComputer:
    def __init__(self, config: IntegratedTrainingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def compute_comprehensive_reward(self, generated_text: str, reference_text: str, 
                                   perplexity: float) -> Dict[str, float]:
        """Compute comprehensive reward signal"""
        rewards = {}
        
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference_text, generated_text)
        rouge_reward = np.mean([score.fmeasure for score in rouge_scores.values()])
        rewards['rouge'] = rouge_reward
        
        # Coherence (simple heuristic based on repetition)
        words = generated_text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        coherence_reward = min(unique_ratio * 2, 1.0)  # Cap at 1.0
        rewards['coherence'] = coherence_reward
        
        # Diversity (vocabulary richness)
        vocab_size = len(set(words))
        diversity_reward = min(vocab_size / 50.0, 1.0)  # Normalize
        rewards['diversity'] = diversity_reward
        
        # Perplexity-based reward (lower is better)
        perplexity_reward = max(0, 1.0 - (perplexity - 1.0) / 10.0)
        rewards['perplexity'] = perplexity_reward
        
        # Weighted total reward
        total_reward = (
            self.config.rouge_weight * rouge_reward +
            self.config.coherence_weight * coherence_reward +
            self.config.diversity_weight * diversity_reward +
            self.config.perplexity_weight * perplexity_reward
        )
        
        rewards['total'] = total_reward
        return rewards

class NeuroPlasticCustomLLM(nn.Module):
    def __init__(self, config: IntegratedTrainingConfig):
        super().__init__()
        self.config = config
        self.base_model = CustomLanguageModel(config)
        self.plasticity = NeuroPlasticityMechanism(self.base_model, config)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> CausalLMOutput:
        return self.base_model(input_ids, attention_mask, labels)
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                 num_return_sequences: int = 1, temperature: float = 0.7, 
                 do_sample: bool = True, pad_token_id: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Generate method with proper encapsulation"""
        return self.base_model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            **kwargs
        )
    
    def apply_plasticity_updates(self, reward_signal: float):
        """Apply neuroplasticity-based parameter updates"""
        modulations = self.plasticity.compute_plasticity_modulation(reward_signal)
        
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in modulations and param.requires_grad:
                    param.data += self.config.plasticity_lr * modulations[name]

class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, num_samples: int = 10000, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Expanded synthetic data templates
        self.templates = [
            "The concept of {} relates to {} in that {}. This connection demonstrates {}.",
            "When analyzing {}, we must consider {} because {}. Furthermore, {}.",
            "In the context of {}, {} plays a crucial role by {}. This is evidenced by {}.",
            "The relationship between {} and {} is complex, involving {}. Specifically, {}.",
            "Understanding {} requires examining {} from the perspective of {}. This reveals {}.",
            "The phenomenon of {} can be explained through {}, which shows that {}. Additionally, {}.",
            "Researchers have found that {} influences {} by {}. This finding suggests {}.",
            "The interaction between {} and {} results in {}, demonstrating that {}.",
        ]
        
        self.concepts = [
            "machine learning", "artificial intelligence", "neural networks", "deep learning",
            "natural language processing", "computer vision", "data science", "algorithms",
            "quantum computing", "robotics", "automation", "cybersecurity", "blockchain",
            "cloud computing", "big data", "statistics", "mathematics", "physics",
            "chemistry", "biology", "psychology", "philosophy", "economics", "sociology"
        ]
        
        self.samples = self._generate_samples(num_samples)
    
    def _generate_samples(self, num_samples: int) -> List[str]:
        samples = []
        for _ in range(num_samples):
            template = random.choice(self.templates)
            concepts_sample = random.sample(self.concepts, 4)
            text = template.format(*concepts_sample)
            samples.append(text)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels with proper padding token handling
        labels = input_ids.clone()
        # Set padding tokens to -100 so they're ignored in loss calculation
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': text
        }

class NeuroplasticTrainingOrchestrator:
    def __init__(self, config: IntegratedTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.reward_computer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_reward = 0.0
        self.curriculum_level = 1
        
        # Long-term performance tracking for plasticity
        self.performance_history = []  # Store recent evaluation results
        self.performance_ema = None    # Exponential moving average of performance
        self.performance_trend = 0.0   # Track improvement trend
        self.last_plasticity_update_step = 0
        
    def initialize_components(self):
        """Initialize all training components"""
        logger.info("Initializing training components...")
        
        # Initialize tokenizer (using GPT-2 tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Adjust config vocab size to match tokenizer
        self.config.vocab_size = self.tokenizer.vocab_size
        logger.info(f"Adjusted vocab_size to {self.config.vocab_size}")
        
        # Initialize model
        self.model = NeuroPlasticCustomLLM(self.config).to(self.device)
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
        
        # Initialize reward computer
        self.reward_computer = AdvancedRewardComputer(self.config, self.tokenizer)
        
        # Create data loaders
        train_dataset = SyntheticDataset(self.tokenizer, num_samples=50000, max_length=512)
        val_dataset = SyntheticDataset(self.tokenizer, num_samples=5000, max_length=512)
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Calculate training steps dynamically
        num_training_steps = self.config.num_epochs * len(self.train_loader)
        warmup_steps = int(self.config.warmup_steps_ratio * num_training_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info("All components initialized successfully!")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Collect gradients for plasticity
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            # Update plasticity traces with gradients (but don't apply updates yet)
            self.model.plasticity.update_plasticity_traces(gradients)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Evaluation and checkpointing
            if self.global_step % self.config.eval_frequency == 0:
                val_loss, val_reward = self.evaluate()
                
                # Apply plasticity updates based on long-term performance trends
                self._update_plasticity_based_on_performance(val_reward)
                
                self.model.train()  # Return to training mode
            
            if self.global_step % self.config.save_frequency == 0:
                self.save_checkpoint()
        
        return total_loss / num_batches
    
    def evaluate(self):
        """Comprehensive evaluation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        logger.info("Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        # Generate sample text for quality evaluation using dynamic references
        sample_rewards = []
        
        # Use a few validation samples for more meaningful ROUGE evaluation
        eval_samples = []
        for i, batch in enumerate(self.val_loader):
            if i >= 5:  # Limit to 5 batches for efficiency
                break
            eval_samples.extend([(batch['input_ids'][j], batch['text'][j]) for j in range(min(2, len(batch['text'])))])
        
        for input_ids_sample, reference_text in eval_samples[:5]:  # Evaluate 5 samples
            # Use first part of the reference as prompt
            words = reference_text.split()
            prompt_words = words[:len(words)//3]  # Use first third as prompt
            prompt = ' '.join(prompt_words)
            
            # The remaining text becomes our reference for ROUGE
            reference_continuation = ' '.join(words[len(words)//3:])
            
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids, max_length=input_ids.shape[1] + 50, num_return_sequences=1,
                    temperature=0.7, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # Extract only the generated continuation (remove prompt)
            generated_continuation = generated_text[len(prompt):].strip()
            
            rewards = self.reward_computer.compute_comprehensive_reward(
                generated_continuation, reference_continuation, perplexity
            )
            sample_rewards.append(rewards['total'])
        
        avg_reward = np.mean(sample_rewards)
        
        # Update curriculum based on reward
        if avg_reward > self.config.curriculum_threshold and avg_reward > self.best_reward + self.config.min_loss_improvement:
            self.curriculum_level += 1
            self.best_reward = avg_reward
            logger.info(f"Curriculum advanced to level {self.curriculum_level}")
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, Avg Reward: {avg_reward:.4f}")
        
        return avg_loss, avg_reward
    
    def _update_plasticity_based_on_performance(self, current_reward: float):
        """Update neuroplasticity based on long-term performance trends"""
        # Update performance history
        self.performance_history.append(current_reward)
        
        # Keep only recent history (last 10 evaluations)
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        # Update exponential moving average of performance
        alpha = 0.3  # EMA decay factor
        if self.performance_ema is None:
            self.performance_ema = current_reward
        else:
            self.performance_ema = alpha * current_reward + (1 - alpha) * self.performance_ema
        
        # Calculate performance trend (improvement over time)
        if len(self.performance_history) >= 3:
            # Compare recent performance to earlier performance
            recent_avg = np.mean(self.performance_history[-3:])
            earlier_avg = np.mean(self.performance_history[:-3]) if len(self.performance_history) > 3 else self.performance_history[0]
            self.performance_trend = recent_avg - earlier_avg
        
        # Determine plasticity signal based on long-term trends
        plasticity_signal = self._compute_plasticity_signal(current_reward)
        
        # Apply plasticity updates
        if abs(plasticity_signal) > 0.1:  # Only update if signal is meaningful
            logger.info(f"Applying plasticity update with signal: {plasticity_signal:.4f}")
            self.model.apply_plasticity_updates(plasticity_signal)
            self.last_plasticity
            # Continuation from the unfinished _update_plasticity_based_on_performance method

            self.last_plasticity_update_step = self.global_step
    
    def _compute_plasticity_signal(self, current_reward: float) -> float:
        """Compute neuroplasticity signal based on performance patterns"""
        # Base signal from current reward relative to moving average
        base_signal = current_reward - (self.performance_ema or 0.5)
        
        # Enhance signal based on performance trend
        trend_bonus = self.performance_trend * 0.5
        
        # Add exploration factor if performance has plateaued
        if len(self.performance_history) >= 5:
            recent_std = np.std(self.performance_history[-5:])
            if recent_std < 0.02:  # Performance has plateaued
                exploration_signal = 0.3  # Encourage exploration
            else:
                exploration_signal = 0.0
        else:
            exploration_signal = 0.0
        
        # Combine signals with adaptive weighting
        total_signal = base_signal + trend_bonus + exploration_signal
        
        # Clip to reasonable range
        return np.clip(total_signal, -1.0, 1.0)
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'plasticity_traces': self.model.plasticity.plasticity_traces,
            'momentum_buffers': self.model.plasticity.momentum_buffers,
            'low_rank_components': self.model.plasticity.low_rank_components,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'curriculum_level': self.curriculum_level,
            'performance_history': self.performance_history,
            'performance_ema': self.performance_ema,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_step_{self.global_step}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved at step {self.global_step}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore plasticity components
        self.model.plasticity.plasticity_traces = checkpoint['plasticity_traces']
        self.model.plasticity.momentum_buffers = checkpoint['momentum_buffers']
        self.model.plasticity.low_rank_components = checkpoint['low_rank_components']
        
        # Restore training state
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']
        self.best_reward = checkpoint['best_reward']
        self.curriculum_level = checkpoint['curriculum_level']
        self.performance_history = checkpoint['performance_history']
        self.performance_ema = checkpoint['performance_ema']
        
        logger.info(f"Checkpoint loaded from step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting neuroplastic training...")
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Train one epoch
                train_loss = self.train_epoch()
                
                # End-of-epoch evaluation
                val_loss, val_reward = self.evaluate()
                
                logger.info(f"Epoch {epoch + 1} completed - Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Reward: {val_reward:.4f}")
                
                # Save epoch checkpoint
                self.save_checkpoint()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint()
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.save_checkpoint()
            raise

# PHASE 1 VALIDATION: Neuroplasticity Effectiveness Test
class NeuroplasticityValidator:
    """Validate that neuroplasticity mechanism provides actual benefits"""
    
    def __init__(self, config: IntegratedTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def validate_plasticity_benefit(self) -> Dict[str, float]:
        """Compare neuroplastic model against baseline on same tasks"""
        logger.info("Validating neuroplasticity benefits...")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.config.vocab_size = tokenizer.vocab_size
        
        # Create models
        baseline_model = CustomLanguageModel(self.config).to(self.device)
        neuroplastic_model = NeuroPlasticCustomLLM(self.config).to(self.device)
        
        # Create test dataset
        test_dataset = SyntheticDataset(tokenizer, num_samples=1000, max_length=256)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Train both models for limited steps
        results = {}
        
        for model_name, model in [("baseline", baseline_model), ("neuroplastic", neuroplastic_model)]:
            logger.info(f"Training {model_name} model...")
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            model.train()
            
            total_loss = 0.0
            num_steps = 500  # Limited training for validation
            
            for step, batch in enumerate(test_loader):
                if step >= num_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if model_name == "neuroplastic":
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    # Apply plasticity updates periodically
                    if step > 0 and step % 50 == 0:
                        # Simple reward based on loss improvement
                        reward = -loss.item()  # Lower loss = higher reward
                        model.apply_plasticity_updates(reward * 0.1)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / min(num_steps, len(test_loader))
            perplexity = math.exp(avg_loss)
            
            results[model_name] = {
                'final_loss': avg_loss,
                'final_perplexity': perplexity
            }
            
            logger.info(f"{model_name} - Final Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Calculate improvement
        baseline_ppl = results['baseline']['final_perplexity']
        neuroplastic_ppl = results['neuroplastic']['final_perplexity']
        improvement = (baseline_ppl - neuroplastic_ppl) / baseline_ppl * 100
        
        results['improvement_percentage'] = improvement
        
        logger.info(f"Neuroplasticity improvement: {improvement:.2f}%")
        
        if improvement < 2.0:
            logger.warning("Neuroplasticity shows minimal benefit (<2%). Consider revising mechanism.")
        elif improvement >= 5.0:
            logger.info("Neuroplasticity shows significant benefit (≥5%). Mechanism validated!")
        
        return results

# SCALED ARCHITECTURE: Modern 7B Parameter Model
@dataclass
class ScaledModelConfig:
    # Scaled architecture (Llama-2-7B inspired)
    vocab_size: int = 32000
    embed_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    ffn_dim: int = 11008
    max_seq_length: int = 32768  # Extended context
    dropout: float = 0.0  # Minimal dropout for large models
    
    # RoPE parameters
    rope_theta: float = 10000.0
    
    # Training parameters (adjusted for scale)
    batch_size: int = 2  # Reduced for memory
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    num_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps_ratio: float = 0.03

class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, embed_dim: int, max_seq_length: int, theta: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.theta = theta
        
        # Precompute rotation matrices
        inv_freq = 1.0 / (theta ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cached_seq_len = None
        self._cached_cos = None
        self._cached_sin = None
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate cos and sin embeddings for given sequence length"""
        if seq_len != self._cached_seq_len:
            # Generate position indices
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            # Compute outer product
            freqs = torch.outer(t, self.inv_freq)
            
            # Duplicate for complex representation
            freqs = torch.cat([freqs, freqs], dim=-1)
            
            cos = freqs.cos()
            sin = freqs.sin()
            
            # Cache results
            self._cached_seq_len = seq_len
            self._cached_cos = cos
            self._cached_sin = sin
        
        return self._cached_cos, self._cached_sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to input tensor"""
    # Split into even and odd dimensions
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    # Apply rotation
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

class ScaledMultiHeadAttention(nn.Module):
    """Scaled attention with RoPE and optimizations"""
    
    def __init__(self, config: ScaledModelConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        assert self.embed_dim % self.num_heads == 0
        
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        
        # RoPE
        self.rope = RoPEEmbedding(self.head_dim, config.max_seq_length, config.rope_theta)
    
    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply RoPE to q and k
        cos, sin = self.rope(T, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

# EVALUATION BENCHMARK SUITE
class BenchmarkEvaluator:
    """Comprehensive evaluation on standard benchmarks"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def evaluate_mmlu_sample(self) -> float:
        """Simplified MMLU-style evaluation"""
        # Sample MMLU-style questions
        questions = [
            {
                "question": "What is the primary function of mitochondria in cells?",
                "choices": ["A) Protein synthesis", "B) Energy production", "C) DNA replication", "D) Waste removal"],
                "answer": "B"
            },
            {
                "question": "Which programming paradigm emphasizes immutable data and functions without side effects?",
                "choices": ["A) Object-oriented", "B) Procedural", "C) Functional", "D) Logic"],
                "answer": "C"
            }
        ]
        
        correct = 0
        total = len(questions)
        
        for q in questions:
            prompt = f"Question: {q['question']}\n{' '.join(q['choices'])}\nAnswer:"
            
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get probabilities for A, B, C, D tokens
                choice_tokens = [self.tokenizer.encode(c)[-1] for c in ['A', 'B', 'C', 'D']]
                choice_probs = F.softmax(logits[choice_tokens], dim=0)
                predicted_idx = torch.argmax(choice_probs).item()
                predicted_answer = ['A', 'B', 'C', 'D'][predicted_idx]
                
                if predicted_answer == q['answer']:
                    correct += 1
        
        return correct / total

    def evaluate_hellaswag_sample(self) -> float:
        """Simplified HellaSwag-style evaluation"""
        scenarios = [
            {
                "context": "A person is cooking pasta in the kitchen.",
                "choices": [
                    "They add salt to the boiling water.",
                    "They start driving to work.",
                    "They begin reading a book.",
                    "They start painting the walls."
                ],
                "answer": 0
            }
        ]
        
        correct = 0
        
        for scenario in scenarios:
            context = scenario['context']
            best_score = float('-inf')
            best_choice = -1
            
            for i, choice in enumerate(scenario['choices']):
                text = f"{context} {choice}"
                input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    score = -outputs.loss.item()  # Negative loss as score
                
                if score > best_score:
                    best_score = score
                    best_choice = i
            
            if best_choice == scenario['answer']:
                correct += 1
        
        return correct / len(scenarios)

# MAIN EXECUTION WITH PHASE 1 VALIDATION
def main():
    """Main execution with validation and scaled training"""
    
    # Phase 1: Validate neuroplasticity mechanism
    logger.info("Phase 1: Validating neuroplasticity mechanism...")
    
    config = IntegratedTrainingConfig()
    validator = NeuroplasticityValidator(config)
    
    try:
        validation_results = validator.validate_plasticity_benefit()
        
        improvement = validation_results['improvement_percentage']
        
        if improvement >= 5.0:
            logger.info(f"✅ Neuroplasticity validated with {improvement:.1f}% improvement!")
            proceed_with_scaling = True
        elif improvement >= 2.0:
            logger.info(f"⚠️ Neuroplasticity shows modest improvement ({improvement:.1f}%). Proceeding with caution.")
            proceed_with_scaling = True
        else:
            logger.warning(f"❌ Neuroplasticity shows minimal benefit ({improvement:.1f}%). Consider algorithm revision.")
            proceed_with_scaling = False
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        proceed_with_scaling = False
    
    if not proceed_with_scaling:
        logger.info("Stopping execution due to validation failure. Please revise the neuroplasticity mechanism.")
        return
    
    # Phase 2: Scale up and train
    logger.info("Phase 2: Scaling up for production training...")
    
    # Use original config for now (scaled config would require more resources)
    training_config = IntegratedTrainingConfig()
    training_config.num_epochs = 3  # Reduced for demonstration
    
    orchestrator = NeuroplasticTrainingOrchestrator(training_config)
    
    try:
        orchestrator.initialize_components()
        orchestrator.train()
        
        logger.info("Training completed successfully!")
        
        # Phase 3: Quick benchmark evaluation
        logger.info("Phase 3: Running benchmark evaluation...")
        
        evaluator = BenchmarkEvaluator(orchestrator.model, orchestrator.tokenizer)
        
        mmlu_score = evaluator.evaluate_mmlu_sample()
        hellaswag_score = evaluator.evaluate_hellaswag_sample()
        
        logger.info(f"Benchmark Results:")
        logger.info(f"MMLU Sample Score: {mmlu_score:.1%}")
        logger.info(f"HellaSwag Sample Score: {hellaswag_score:.1%}")
        
        # Save final model
        orchestrator.save_checkpoint()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()
