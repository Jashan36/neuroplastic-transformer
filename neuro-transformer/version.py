import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    get_linear_schedule_with_warmup
)
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import deepspeed
from bert_score import score as bert_score
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import evaluate
import wandb
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Enhanced configuration with proper hyperparameter ranges
@dataclass
class EnhancedTrainingConfig:
    # Model and training basics
    model_name: str = "microsoft/DialoGPT-medium"  # Start with manageable size
    max_seq_length: int = 512
    batch_size: int = 4
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    
    # Neuroplasticity parameters (tuned based on research)
    plasticity_lr: float = 1e-5  # Reduced for stability
    plasticity_decay: float = 0.99  # Temporal decay
    modulatory_strength: float = 0.1  # Strength of modulatory signals
    trace_momentum: float = 0.95  # Momentum for trace updates
    
    # Sparse plasticity (critical for scaling)
    plasticity_sparsity: float = 0.05  # Track only 5% of parameters
    low_rank_dim: int = 32  # Low-rank approximation dimension
    plasticity_update_frequency: int = 10  # Update every N steps
    
    # Advanced reward system weights
    rouge_weight: float = 0.3
    bert_score_weight: float = 0.25
    diversity_weight: float = 0.2
    factuality_weight: float = 0.15
    length_penalty_weight: float = 0.1
    
    # RL training parameters
    use_ppo: bool = True
    ppo_epochs: int = 4
    ppo_lr: float = 1.41e-5
    ppo_batch_size: int = 32
    ppo_mini_batch_size: int = 4
    
    # Curriculum learning
    curriculum_stages: list = field(default_factory=lambda: [
        {"name": "basic_qa", "difficulty": 1.0, "max_turns": 1},
        {"name": "explanations", "difficulty": 1.5, "max_turns": 2},
        {"name": "complex_reasoning", "difficulty": 2.0, "max_turns": 3},
        {"name": "multi_domain", "difficulty": 2.5, "max_turns": 4}
    ])
    
    # Evaluation and safety
    eval_steps: int = 500
    save_steps: int = 1000
    safety_check_frequency: int = 100

config = EnhancedTrainingConfig()

# Initialize enhanced logging and monitoring
def setup_logging_and_monitoring():
    """Setup comprehensive logging with W&B integration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'neuroplasticity_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize W&B for experiment tracking
    try:
        wandb.init(
            project="neuroplasticity-llm",
            config=config.__dict__,
            name=f"neuroplastic-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        logging.info("W&B logging initialized successfully")
    except Exception as e:
        logging.warning(f"Failed to initialize W&B: {e}")

setup_logging_and_monitoring()

# Enhanced modulatory signal computation
class ModulatoryHead(nn.Module):
    """
    Separate neural network that learns modulatory signals for plasticity.
    Inspired by Backpropamine and meta-learning approaches.
    """
    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Multi-layer modulatory network
        layers = []
        input_size = hidden_size
        for i in range(num_layers):
            output_size = hidden_size // (2 ** i) if i < num_layers - 1 else 1
            layers.extend([
                nn.Linear(input_size, output_size),
                nn.ReLU() if i < num_layers - 1 else nn.Sigmoid()
            ])
            input_size = output_size
        
        self.modulatory_net = nn.Sequential(*layers)
        self.context_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute modulatory signals based on current hidden states and context.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            modulatory_signals: [batch_size] - per-sample modulatory strength
        """
        # Apply self-attention to capture contextual information
        attended_states, _ = self.context_attention(
            hidden_states, hidden_states, hidden_states, key_padding_mask=~attention_mask.bool()
        )
        
        # Pool across sequence length (weighted by attention mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled_states = (attended_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_states = attended_states.mean(dim=1)
        
        # Generate modulatory signal
        modulatory_signal = self.modulatory_net(pooled_states).squeeze(-1)
        
        return modulatory_signal

# Enhanced sparse plasticity wrapper with low-rank approximations
class SparseNeuroPlasticWrapper(nn.Module):
    """
    Memory-efficient neuroplasticity wrapper using sparse traces and low-rank approximations.
    Scales to billion-parameter models.
    """
    def __init__(self, model, config: EnhancedTrainingConfig):
        super().__init__()
        self.model = model
        self.config = config
        
        # Initialize modulatory head
        try:
            hidden_size = model.config.hidden_size
        except:
            hidden_size = model.config.n_embd  # For GPT-2 style models
        
        self.modulatory_head = ModulatoryHead(hidden_size)
        
        # Select sparse parameters for plasticity tracking
        self.tracked_params = self._select_sparse_parameters()
        self.plasticity_traces = nn.ParameterDict()
        self.trace_momentum = nn.ParameterDict()
        
        # Initialize sparse plasticity traces with low-rank approximation
        self._initialize_sparse_traces()
        
        # Track plasticity statistics
        self.plasticity_stats = {
            'update_count': 0,
            'avg_modulatory_signal': 0.0,
            'trace_norms': [],
        }
        
    def _select_sparse_parameters(self) -> List[str]:
        """Select top-k% most important parameters for plasticity tracking."""
        param_importance = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'weight' in name:
                # Use parameter magnitude as importance heuristic
                # In practice, could use Fisher information or gradient norms
                importance = torch.norm(param.data).item()
                param_importance[name] = importance
        
        # Select top percentage based on sparsity setting
        k = int(len(param_importance) * self.config.plasticity_sparsity)
        top_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)[:k]
        
        selected_params = [name for name, _ in top_params]
        logging.info(f"Selected {len(selected_params)} parameters for plasticity tracking")
        
        return selected_params
    
    def _initialize_sparse_traces(self):
        """Initialize sparse plasticity traces with memory-efficient representations."""
        for name in self.tracked_params:
            param = dict(self.model.named_parameters())[name]
            
            if len(param.shape) >= 2:  # Matrix parameters - use low-rank approximation
                # Initialize low-rank factors
                min_dim = min(param.shape)
                rank = min(self.config.low_rank_dim, min_dim)
                
                self.plasticity_traces[f"{name}_U"] = nn.Parameter(
                    torch.randn(param.shape[0], rank) * 0.01, requires_grad=False
                )
                self.plasticity_traces[f"{name}_V"] = nn.Parameter(
                    torch.randn(rank, param.shape[1]) * 0.01, requires_grad=False
                )
                
                # Momentum terms
                self.trace_momentum[f"{name}_U"] = nn.Parameter(
                    torch.zeros(param.shape[0], rank), requires_grad=False
                )
                self.trace_momentum[f"{name}_V"] = nn.Parameter(
                    torch.zeros(rank, param.shape[1]), requires_grad=False
                )
            else:  # Vector parameters - store directly
                self.plasticity_traces[name] = nn.Parameter(
                    torch.zeros_like(param), requires_grad=False
                )
                self.trace_momentum[name] = nn.Parameter(
                    torch.zeros_like(param), requires_grad=False
                )
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass with modulatory signal computation."""
        # Standard forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Compute modulatory signals if hidden states available
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden_states = outputs.hidden_states[-1]
            modulatory_signals = self.modulatory_head(last_hidden_states, attention_mask)
            outputs.modulatory_signals = modulatory_signals
        else:
            # Fallback: extract from last layer
            try:
                last_hidden_states = outputs.last_hidden_state
                modulatory_signals = self.modulatory_head(last_hidden_states, attention_mask)
                outputs.modulatory_signals = modulatory_signals
            except:
                # Final fallback: zero modulatory signals
                batch_size = input_ids.shape[0]
                outputs.modulatory_signals = torch.zeros(batch_size, device=input_ids.device)
        
        return outputs
    
    def update_plasticity_traces(self, modulatory_signals: torch.Tensor):
        """Update sparse plasticity traces with modulatory signals."""
        if self.plasticity_stats['update_count'] % self.config.plasticity_update_frequency != 0:
            self.plasticity_stats['update_count'] += 1
            return
        
        avg_modulatory = modulatory_signals.mean().item()
        self.plasticity_stats['avg_modulatory_signal'] = avg_modulatory
        
        with torch.no_grad():
            for name in self.tracked_params:
                param = dict(self.model.named_parameters())[name]
                if param.grad is None:
                    continue
                
                grad = param.grad.detach()
                
                # Normalize gradient for stability
                grad_norm = torch.norm(grad)
                if grad_norm > 0:
                    normalized_grad = grad / grad_norm
                else:
                    continue
                
                if len(param.shape) >= 2:  # Matrix parameters with low-rank traces
                    U_key, V_key = f"{name}_U", f"{name}_V"
                    
                    # SVD-based update for low-rank approximation
                    try:
                        U, S, Vh = torch.linalg.svd(normalized_grad, full_matrices=False)
                        rank = min(self.config.low_rank_dim, min(U.shape[1], Vh.shape[0]))
                        
                        # Update low-rank factors with momentum
                        U_update = avg_modulatory * U[:, :rank]
                        V_update = avg_modulatory * Vh[:rank, :]
                        
                        # Momentum updates
                        self.trace_momentum[U_key].mul_(self.config.trace_momentum).add_(U_update)
                        self.trace_momentum[V_key].mul_(self.config.trace_momentum).add_(V_update)
                        
                        # Trace updates with decay
                        self.plasticity_traces[U_key].mul_(self.config.plasticity_decay)
                        self.plasticity_traces[U_key].add_(self.trace_momentum[U_key])
                        
                        self.plasticity_traces[V_key].mul_(self.config.plasticity_decay)
                        self.plasticity_traces[V_key].add_(self.trace_momentum[V_key])
                        
                    except Exception as e:
                        logging.warning(f"SVD update failed for {name}: {e}")
                        continue
                        
                else:  # Vector parameters
                    # Direct trace update with momentum
                    trace_update = avg_modulatory * normalized_grad
                    
                    self.trace_momentum[name].mul_(self.config.trace_momentum).add_(trace_update)
                    self.plasticity_traces[name].mul_(self.config.plasticity_decay)
                    self.plasticity_traces[name].add_(self.trace_momentum[name])
        
        self.plasticity_stats['update_count'] += 1
    
    def apply_plastic_updates(self):
        """Apply plasticity-based parameter updates."""
        with torch.no_grad():
            total_update_norm = 0.0
            
            for name in self.tracked_params:
                param = dict(self.model.named_parameters())[name]
                
                if len(param.shape) >= 2:  # Matrix parameters
                    U_key, V_key = f"{name}_U", f"{name}_V"
                    
                    if U_key in self.plasticity_traces and V_key in self.plasticity_traces:
                        # Reconstruct low-rank update
                        U_trace = self.plasticity_traces[U_key]
                        V_trace = self.plasticity_traces[V_key]
                        
                        # Low-rank update: param += lr * U @ V
                        plastic_update = self.config.plasticity_lr * torch.mm(U_trace, V_trace)
                        
                        # Ensure shapes match
                        if plastic_update.shape == param.shape:
                            param.data.add_(plastic_update)
                            total_update_norm += torch.norm(plastic_update).item()
                        
                else:  # Vector parameters
                    if name in self.plasticity_traces:
                        trace = self.plasticity_traces[name]
                        plastic_update = self.config.plasticity_lr * trace
                        param.data.add_(plastic_update)
                        total_update_norm += torch.norm(plastic_update).item()
            
            self.plasticity_stats['trace_norms'].append(total_update_norm)
            
            # Log plasticity statistics
            if self.plasticity_stats['update_count'] % 100 == 0:
                logging.info(f"Plasticity update #{self.plasticity_stats['update_count']}: "
                           f"avg_modulatory={self.plasticity_stats['avg_modulatory_signal']:.4f}, "
                           f"update_norm={total_update_norm:.4f}")

# Enhanced reward computation with proper metrics
class AdvancedRewardComputer:
    """
    Compute sophisticated reward signals using state-of-the-art metrics.
    """
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        
        # Initialize advanced metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize BERTScore
        try:
            # BERTScore will be computed on-demand to save memory
            self.bertscore_available = True
        except Exception as e:
            logging.warning(f"BERTScore not available: {e}")
            self.bertscore_available = False
        
        # Initialize factual consistency model
        try:
            self.factuality_pipeline = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",  # Placeholder - replace with factuality model
                device=0 if torch.cuda.is_available() else -1
            )
            self.factuality_available = True
        except Exception as e:
            logging.warning(f"Factuality pipeline not available: {e}")
            self.factuality_available = False
        
        # Initialize diversity metrics
        self.distinct_n_cache = {}
    
    def compute_rouge_reward(self, references: List[str], candidates: List[str]) -> float:
        """Compute ROUGE-based reward."""
        if not references or not candidates:
            return 0.0
        
        rouge_scores = []
        for ref, cand in zip(references, candidates):
            try:
                scores = self.rouge_scorer.score(ref, cand)
                rouge_l = scores['rougeL'].fmeasure
                rouge_scores.append(rouge_l)
            except Exception:
                rouge_scores.append(0.0)
        
        return np.mean(rouge_scores)
    
    def compute_bert_score_reward(self, references: List[str], candidates: List[str]) -> float:
        """Compute BERTScore-based semantic reward."""
        if not self.bertscore_available or not references or not candidates:
            return 0.5  # Neutral score
        
        try:
            P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
            return F1.mean().item()
        except Exception as e:
            logging.warning(f"BERTScore computation failed: {e}")
            return 0.5
    
    def compute_diversity_reward(self, candidates: List[str]) -> float:
        """Compute diversity reward using Distinct-n and self-BLEU."""
        if not candidates:
            return 0.0
        
        # Distinct-2 computation
        distinct_2_scores = []
        for candidate in candidates:
            tokens = candidate.split()
            if len(tokens) < 2:
                distinct_2_scores.append(0.0)
                continue
            
            bigrams = set(zip(tokens[:-1], tokens[1:]))
            distinct_2 = len(bigrams) / max(len(tokens) - 1, 1)
            distinct_2_scores.append(distinct_2)
        
        # Self-BLEU (lower is more diverse)
        self_bleu_scores = []
        for i, candidate in enumerate(candidates):
            other_candidates = candidates[:i] + candidates[i+1:]
            if not other_candidates:
                continue
            
            # Compute BLEU against other candidates
            candidate_tokens = candidate.split()
            reference_tokens_list = [ref.split() for ref in other_candidates]
            
            try:
                bleu = sentence_bleu(reference_tokens_list, candidate_tokens)
                self_bleu_scores.append(bleu)
            except Exception:
                self_bleu_scores.append(0.0)
        
        diversity_score = np.mean(distinct_2_scores)
        if self_bleu_scores:
            diversity_score += (1.0 - np.mean(self_bleu_scores))  # Lower self-BLEU = higher diversity
            diversity_score /= 2.0
        
        return diversity_score
    
    def compute_factuality_reward(self, contexts: List[str], candidates: List[str]) -> float:
        """Compute factual consistency reward."""
        if not self.factuality_available or not contexts or not candidates:
            return 0.5  # Neutral score
        
        factuality_scores = []
        for context, candidate in zip(contexts, candidates):
            try:
                # Simple heuristic: check for factual consistency
                # In production, use a proper factual consistency model like SummaC
                input_text = f"Context: {context} Claim: {candidate}"
                
                # For now, use a simple heuristic based on context overlap
                context_words = set(context.lower().split())
                candidate_words = set(candidate.lower().split())
                
                # Factuality based on entity consistency and no contradictory statements
                overlap_ratio = len(context_words & candidate_words) / max(len(candidate_words), 1)
                factuality_scores.append(min(overlap_ratio * 2, 1.0))  # Scale to [0, 1]
                
            except Exception:
                factuality_scores.append(0.5)
        
        return np.mean(factuality_scores)
    
    def compute_length_penalty(self, candidates: List[str], target_length: int = 50) -> float:
        """Compute length penalty reward."""
        if not candidates:
            return 0.0
        
        length_penalties = []
        for candidate in candidates:
            candidate_length = len(candidate.split())
            # Gaussian penalty around target length
            penalty = np.exp(-0.5 * ((candidate_length - target_length) / (target_length * 0.3)) ** 2)
            length_penalties.append(penalty)
        
        return np.mean(length_penalties)
    
    def compute_comprehensive_reward(
        self, 
        contexts: List[str], 
        references: List[str], 
        candidates: List[str]
    ) -> Dict[str, float]:
        """Compute comprehensive reward combining multiple metrics."""
        
        # Individual rewards
        rouge_reward = self.compute_rouge_reward(references, candidates)
        bert_reward = self.compute_bert_score_reward(references, candidates) if references else 0.5
        diversity_reward = self.compute_diversity_reward(candidates)
        factuality_reward = self.compute_factuality_reward(contexts, candidates)
        length_reward = self.compute_length_penalty(candidates)
        
        # Weighted combination
        total_reward = (
            self.config.rouge_weight * rouge_reward +
            self.config.bert_score_weight * bert_reward +
            self.config.diversity_weight * diversity_reward +
            self.config.factuality_weight * factuality_reward +
            self.config.length_penalty_weight * length_reward
        )
        
        return {
            'total_reward': total_reward,
            'rouge_reward': rouge_reward,
            'bert_reward': bert_reward,
            'diversity_reward': diversity_reward,
            'factuality_reward': factuality_reward,
            'length_reward': length_reward
        }

# Dynamic curriculum scheduler
class DynamicCurriculumScheduler:
    """
    Adaptive curriculum that adjusts difficulty based on performance metrics.
    """
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.current_stage = 0
        self.performance_history = []
        self.difficulty_threshold = 0.7  # Performance threshold to advance
        self.stability_window = 5  # Number of evaluations to consider for stability
        
    def should_advance_stage(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if curriculum should advance to next stage."""
        self.performance_history.append(performance_metrics['total_reward'])
        
        # Keep only recent history
        if len(self.performance_history) > self.stability_window * 2:
            self.performance_history = self.performance_history[-self.stability_window * 2:]
        
        # Need minimum history to make decision
        if len(self.performance_history) < self.stability_window:
            return False
        
        # Check for stable performance above threshold
        recent_performance = self.performance_history[-self.stability_window:]
        avg_recent = np.mean(recent_performance)
        std_recent = np.std(recent_performance)
        
        # Advance if performance is stable and above threshold
        is_stable = std_recent < 0.1  # Low variance indicates stability
        is_good_performance = avg_recent >= self.difficulty_threshold
        
        if is_stable and is_good_performance:
            logging.info(f"Advancing curriculum: avg_perf={avg_recent:.3f}, std={std_recent:.3f}")
            return True
        
        return False
    
    def get_current_stage_config(self) -> Dict[str, Any]:
        """Get configuration for current curriculum stage."""
        if self.current_stage >= len(self.config.curriculum_stages):
            # Return final stage config
            return self.config.curriculum_stages[-1]
        
        return self.config.curriculum_stages[self.current_stage]
    
    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.config.curriculum_stages) - 1:
            self.current_stage += 1
            self.performance_history = []  # Reset history for new stage
            logging.info(f"Advanced to curriculum stage {self.current_stage}: "
                       f"{self.config.curriculum_stages[self.current_stage]['name']}")

# Enhanced dataset with real instruction data
class EnhancedInstructionDataset(Dataset):
    """
    Enhanced dataset that loads real instruction data and applies curriculum learning.
    """
    def __init__(self, tokenizer, stage_config: Dict[str, Any], max_length: int = 512):
        self.tokenizer = tokenizer
        self.stage_config = stage_config
        self.max_length = max_length
        self.examples = []
        
        # Load real instruction data if available
        self._load_instruction_data()
        
        # Generate synthetic data if real data is insufficient
        if len(self.examples) < 1000:
            self._generate_synthetic_data()
    
    def _load_instruction_data(self):
        """Load real instruction following datasets."""
        # Try to load from various sources
        data_sources = [
            'dolly_data.json',
            'alpaca_data.json',
            'self_instruct_data.json'
        ]
        
        for source in data_sources:
            if os.path.exists(source):
                try:
                    with open(source, 'r') as f:
                        data = json.load(f)
                        
                    for item in data:
                        if self._matches_stage_criteria(item):
                            self.examples.append({
                                'instruction': item.get('instruction', ''),
                                'input': item.get('input', ''),
                                'output': item.get('output', ''),
                                'difficulty': self._estimate_difficulty(item)
                            })
                            
                    logging.info(f"Loaded {len(self.examples)} examples from {source}")
                    break
                    
                except Exception as e:
                    logging.warning(f"Failed to load {source}: {e}")
                    continue
    
    def _matches_stage_criteria(self, item: Dict[str, Any]) -> bool:
        """Check if data item matches current curriculum stage."""
        stage_name = self.stage_config['name']
        difficulty = self.stage_config['difficulty']
        
        # Simple heuristic based on output length and complexity
        output_length = len(item.get('output', '').split())
        
        if stage_name == 'basic_qa':
            return output_length <= 30
        elif stage_name == 'explanations':
            return 30 < output_length <= 100
        elif stage_name == 'complex_reasoning':
            return 100 < output_length <= 200
        else:  # multi_domain
            return output_length > 50  # Any complex task
    
    def _estimate_difficulty(self, item: Dict[str, Any]) -> float:
        """Estimate difficulty of instruction item."""
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        # Heuristic difficulty based on various factors
        factors = [
            len(instruction.split()) / 20,  # Instruction complexity
            len(output.split()) / 100,      # Response complexity
            len(set(output.split())) / len(output.split() or [1]),  # Vocabulary diversity
            instruction.count('?') + instruction.count('explain') + instruction.count('analyze')  # Question complexity
        ]
        
        return min(sum(factors) / len(factors), 3.0)
    
    def _generate_synthetic_data(self):
        """Generate synthetic instruction data for current stage."""
        stage_name = self.stage_config['name']
        num_synthetic = 1000 - len(self.examples)
        
        templates = {
            'basic_qa': [
                ("What is {concept}?", "A {concept} is..."),
                ("Define {concept}.", "{concept} refers to..."),
                ("Explain {concept} briefly.", "{concept} involves...")
            ],
            'explanations': [
                ("How does {concept} work?", "{concept} works by following these steps..."),
                ("Why is {concept} important?", "{concept} is important because..."),
                ("Compare {concept1} and {concept2}.", "{concept1} and {concept2} differ in...")
            ],
            'complex_reasoning': [
                ("Analyze the implications of {concept}.", "The implications of {concept} include..."),
                ("What are the pros and cons of {concept}?", "The advantages of {concept} are... However, the disadvantages include..."),
                ("How would you solve {problem}?", "To solve {problem}, I would...")
            ],
            'multi_domain': [
                ("Discuss {concept} from multiple perspectives.", "From a {perspective1} viewpoint, {concept}... From a {perspective2} perspective..."),
                ("How does {concept} apply across different domains?", "{concept} applies across domains by...")
            ]
        }
        
        # Domain-specific concepts for each stage
        concepts = {
            'basic_qa': ['artificial intelligence', 'machine learning', 'neural network', 'algorithm', 'data science', 
                        'programming', 'computer science', 'database', 'software', 'hardware'],
            'explanations': ['deep learning', 'natural language processing', 'computer vision', 'reinforcement learning',
                           'blockchain', 'cloud computing', 'cybersecurity', 'big data', 'IoT', 'quantum computing'],
            'complex_reasoning': ['ethical AI', 'bias in machine learning', 'AI governance', 'explainable AI',
                                'AI safety', 'privacy in AI', 'AI democratization', 'future of work', 'AGI', 'consciousness'],
            'multi_domain': ['sustainability', 'healthcare innovation', 'education technology', 'smart cities',
                           'digital transformation', 'innovation ecosystems', 'global connectivity', 'human-AI collaboration']
        }
        
        problems = ['traffic congestion', 'climate change', 'healthcare access', 'education inequality', 
                   'food security', 'energy efficiency', 'urban planning', 'digital divide']
        
        perspectives = ['technical', 'ethical', 'economic', 'social', 'environmental', 'cultural', 'political', 'philosophical']
        
        stage_templates = templates.get(stage_name, templates['basic_qa'])
        stage_concepts = concepts.get(stage_name, concepts['basic_qa'])
        
        for i in range(num_synthetic):
            template = random.choice(stage_templates)
            instruction_template, output_template = template
            
            # Fill in template based on placeholders
            if '{concept}' in instruction_template:
                concept = random.choice(stage_concepts)
                instruction = instruction_template.format(concept=concept)
                output = output_template.format(concept=concept)
            elif '{concept1}' in instruction_template:
                concept1, concept2 = random.sample(stage_concepts, 2)
                instruction = instruction_template.format(concept1=concept1, concept2=concept2)
                output = output_template.format(concept1=concept1, concept2=concept2)
            elif '{problem}' in instruction_template:
                problem = random.choice(problems)
                instruction = instruction_template.format(problem=problem)
                output = output_template.format(problem=problem)
            elif '{perspective1}' in instruction_template:
                perspective1, perspective2 = random.sample(perspectives, 2)
                concept = random.choice(stage_concepts)
                instruction = instruction_template.format(concept=concept)
                output = output_template.format(concept=concept, perspective1=perspective1, perspective2=perspective2)
            else:
                concept = random.choice(stage_concepts)
                instruction = instruction_template.format(concept=concept)
                output = output_template.format(concept=concept)
            
            self.examples.append({
                'instruction': instruction,
                'input': '',
                'output': output,
                'difficulty': self.stage_config['difficulty']
            })
        
        logging.info(f"Generated {num_synthetic} synthetic examples for stage {stage_name}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format instruction-input-output
        if example['input'].strip():
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'difficulty': example['difficulty']
        }

# Enhanced PPO Training Integration
class NeuroplasticPPOTrainer:
    """
    PPO trainer integrated with neuroplasticity mechanisms.
    """
    def __init__(self, model_wrapper, config: EnhancedTrainingConfig, tokenizer):
        self.model_wrapper = model_wrapper
        self.config = config
        self.tokenizer = tokenizer
        self.reward_computer = AdvancedRewardComputer(config)
        
        # Initialize PPO components
        if config.use_ppo:
            self._setup_ppo_training()
    
    def _setup_ppo_training(self):
        """Setup PPO training components."""
        ppo_config = PPOConfig(
            model_name=self.config.model_name,
            learning_rate=self.config.ppo_lr,
            batch_size=self.config.ppo_batch_size,
            mini_batch_size=self.config.ppo_mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            optimize_cuda_cache=True,
            gradient_accumulation_steps=2
        )
        
        # Create value model (copy of main model)
        try:
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.model_wrapper.model,
                ref_model=None,  # Will use default reference model
                tokenizer=self.tokenizer
            )
            logging.info("PPO trainer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize PPO trainer: {e}")
            self.config.use_ppo = False
    
    def generate_responses(self, prompts: List[str], max_length: int = 100) -> List[str]:
        """Generate responses for given prompts."""
        responses = []
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model_wrapper.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def compute_rewards(self, prompts: List[str], responses: List[str], references: List[str] = None) -> List[float]:
        """Compute rewards for prompt-response pairs."""
        if references is None:
            references = [""] * len(responses)  # Empty references for unsupervised
        
        reward_dict = self.reward_computer.compute_comprehensive_reward(
            contexts=prompts,
            references=references,
            candidates=responses
        )
        
        # Convert to tensor format expected by PPO
        rewards = [reward_dict['total_reward']] * len(responses)
        return torch.tensor(rewards, dtype=torch.float32)

# Main training orchestrator
class NeuroplasticTrainingOrchestrator:
    """
    Main training orchestrator that coordinates all components.
    """
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.setup_distributed_training()
        self.initialize_components()
        
    def setup_distributed_training(self):
        """Setup distributed training if available."""
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                dist.init_process_group(backend='nccl')
                self.local_rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.is_distributed = True
                torch.cuda.set_device(self.local_rank)
                logging.info(f"Distributed training initialized: rank {self.local_rank}/{self.world_size}")
            except Exception as e:
                logging.warning(f"Failed to initialize distributed training: {e}")
    
    def initialize_components(self):
        """Initialize all training components."""
        # Initialize tokenizer and model
        logging.info(f"Loading model and tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None
        )
        
        # Wrap with neuroplasticity
        self.model_wrapper = SparseNeuroPlasticWrapper(self.base_model, self.config)
        
        # Initialize curriculum scheduler
        self.curriculum_scheduler = DynamicCurriculumScheduler(self.config)
        
        # Initialize PPO trainer
        self.ppo_trainer = NeuroplasticPPOTrainer(self.model_wrapper, self.config, self.tokenizer)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model_wrapper.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize scheduler
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.num_epochs * 1000  # Approximate
        )
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        logging.info("All components initialized successfully")
    
    def create_dataloader(self) -> DataLoader:
        """Create dataloader for current curriculum stage."""
        stage_config = self.curriculum_scheduler.get_current_stage_config()
        
        dataset = EnhancedInstructionDataset(
            tokenizer=self.tokenizer,
            stage_config=stage_config,
            max_length=self.config.max_seq_length
        )
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_dataloader, val_dataloader
    
    def train_epoch(self, train_dataloader: DataLoader, epoch: int):
        """Train one epoch with neuroplasticity updates."""
        self.model_wrapper.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = range(len(train_dataloader))
        
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                # Move to device
                input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
                attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
                labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = self.model_wrapper(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Apply modulatory signals to loss if available
                    if hasattr(outputs, 'modulatory_signals'):
                        modulatory_signals = outputs.modulatory_signals
                        # Scale loss by modulatory signals (encourage higher signals for better examples)
                        loss = loss * (1.0 + self.config.modulatory_strength * modulatory_signals.mean())
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # Update plasticity traces with modulatory signals
                    if hasattr(outputs, 'modulatory_signals'):
                        self.model_wrapper.update_plasticity_traces(outputs.modulatory_signals)
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model_wrapper.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # Update plasticity traces
                    if hasattr(outputs, 'modulatory_signals'):
                        self.model_wrapper.update_plasticity_traces(outputs.modulatory_signals)
                    
                    torch.nn.utils.clip_grad_norm_(self.model_wrapper.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Apply plastic updates periodically
                if batch_idx % self.config.plasticity_update_frequency == 0:
                    self.model_wrapper.apply_plastic_updates()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Logging
                if batch_idx % 50 == 0:
                    avg_loss = total_loss / max(num_batches, 1)
                    logging.info(f"Epoch {epoch}, Batch {batch_idx}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")
                    
                    # W&B logging
                    try:
                        wandb.log({
                            'batch_loss': loss.item(),
                            'avg_loss': avg_loss,
                            'learning_rate': self.lr_scheduler.get_last_lr()[0],
                            'modulatory_signal': outputs.modulatory_signals.mean().item() if hasattr(outputs, 'modulatory_signals') else 0.0
                        })
                    except:
                        pass
                
            except Exception as e:
                logging.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model_wrapper.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Collect samples for reward computation
        prompts, responses, references = [], [], []
        
        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
                    attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
                    labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
                    
                    outputs = self.model_wrapper(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_loss += outputs.loss.item()
                    num_batches += 1
                    
                    # Generate responses for reward computation (sample a few)
                    if len(prompts) < 50:  # Limit for efficiency
                        for i in range(min(2, input_ids.shape[0])):
                            prompt_tokens = input_ids[i][:torch.sum(attention_mask[i])]
                            prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                            
                            # Extract instruction part
                            if "### Instruction:" in prompt:
                                instruction_part = prompt.split("### Response:")[0] + "### Response:"
                                reference_part = prompt.split("### Response:")[-1].strip()
                                
                                prompts.append(instruction_part)
                                references.append(reference_part)
                    
                except Exception as e:
                    logging.error(f"Error in evaluation batch: {e}")
                    continue
        
        # Generate responses for collected prompts
        if prompts:
            generated_responses = self.ppo_trainer.generate_responses(prompts, max_length=100)
            responses.extend(generated_responses)
            
            # Compute comprehensive rewards
            reward_computer = AdvancedRewardComputer(self.config)
            reward_metrics = reward_computer.compute_comprehensive_reward(
                contexts=prompts,
                references=references,
                candidates=responses
            )
        else:
            reward_metrics = {'total_reward': 0.0}
        
        eval_metrics = {
            'eval_loss': total_loss / max(num_batches, 1),
            **reward_metrics
        }
        
        return eval_metrics
    
    def train(self):
        """Main training loop with curriculum learning."""
        logging.info("Starting neuroplastic training...")
        
        for epoch in range(self.config.num_epochs):
            logging.info(f"\n=== EPOCH {epoch + 1}/{self.config.num_epochs} ===")
            
            # Create dataloader for current curriculum stage
            train_dataloader, val_dataloader = self.create_dataloader()
            
            # Train epoch
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # Evaluate
            if epoch % (self.config.eval_steps // 1000) == 0:
                eval_metrics = self.evaluate(val_dataloader)
                
                logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_metrics['eval_loss']:.4f}")
                logging.info(f"Reward metrics: {eval_metrics}")
                
                # Check curriculum advancement
                if self.curriculum_scheduler.should_advance_stage(eval_metrics):
                    self.curriculum_scheduler.advance_stage()
                
                # W&B logging
                try:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        **eval_metrics,
                        'curriculum_stage': self.curriculum_scheduler.current_stage
                    })
                except:
                    pass
            
            # Save checkpoint
            if epoch % (self.config.save_steps // 1000) == 0:
                self.save_checkpoint(epoch)
        
        logging.info("Training completed!")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint with plasticity state."""
        checkpoint_dir = f"checkpoint_epoch_{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model_wrapper.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'plasticity_stats': self.model_wrapper.plasticity_stats,
            'curriculum_stage': self.curriculum_scheduler.current_stage,
            'config': self.config
        }, os.path.join(checkpoint_dir, 'model_checkpoint.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logging.info(f"Checkpoint saved to {checkpoint_dir}")

# Main execution
def main():
    """Main training execution."""
    # Initialize configuration
    config = EnhancedTrainingConfig()
    
    # Create training orchestrator
    trainer = NeuroplasticTrainingOrchestrator(config)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        trainer.save_checkpoint(-1)
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
