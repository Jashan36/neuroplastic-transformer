# Neuroplastic Language Model: Advanced Transformer with Neuroplasticity

This repository contains a production-ready implementation of a large language model (LLM) with advanced neuroplasticity-inspired training, curriculum learning, and multi-objective reward optimization. The core model is a custom transformer architecture, trained from scratch with a unique neuroplasticity mechanism that adapts learning based on long-term performance trends.

## Key Features
- **Custom Transformer Architecture:** Highly configurable (e.g., 24 layers, 2048-dim embeddings, 16 heads) with efficient attention and feedforward blocks.
- **Neuroplasticity Mechanism:** Inspired by biological learning, the model maintains plasticity traces, momentum buffers, and low-rank modulatory updates to reinforce beneficial learning pathways based on reward signals.
- **Multi-Objective Reward Function:** Training is guided by a comprehensive reward signal combining ROUGE, coherence, diversity, and perplexity metrics.
- **Curriculum Learning:** Training progresses through increasingly complex tasks (definitions, explanations, dialogues) to stabilize and accelerate learning.
- **Benchmark Evaluation:** Includes built-in evaluation on MMLU-style and HellaSwag-style tasks for quick benchmarking.
- **Synthetic Data Generation:** Uses robust synthetic data for rapid prototyping and validation.
- **Checkpointing:** Automatic saving and loading of model, optimizer, scheduler, and plasticity state.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

**Required packages:**
- torch
- transformers
- numpy
- nltk
- datasets
- tqdm
- rouge-score

## Setup and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Prepare Environment
(Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run the Model
To run the full neuroplastic LLM pipeline (validation, training, benchmarking):
```bash
python neuro-transformer/model1.py
```

- The script will first validate the neuroplasticity mechanism against a baseline.
- If validation passes, it will proceed to full training with curriculum learning.
- After training, it runs quick benchmark evaluations and saves checkpoints in `neuroplastic_checkpoints/`.

### 4. Checkpoints and Logs
- Model checkpoints, optimizer state, and plasticity traces are saved in `neuroplastic_checkpoints/`.
- Training logs are saved in `training_logs/`.

## Model Overview (from model1.py)
- **Architecture:** Custom transformer (configurable layers, heads, embedding size)
- **Plasticity:** Hebbian-like updates, momentum, low-rank modulation
- **Reward:** Weighted sum of ROUGE, coherence, diversity, perplexity
- **Curriculum:** Definitions → Explanations → Dialogues
- **Evaluation:** MMLU and HellaSwag samples

## Extending or Customizing
- Adjust model/training config in `IntegratedTrainingConfig` (in `model1.py`).
- Add new reward metrics in `AdvancedRewardComputer`.
- Plug in your own data by modifying the `SyntheticDataset` or loading from files.

## Main Formulas Used in Training

### 1. Language Modeling Loss (Cross-Entropy)
The core loss for next-token prediction:

$$
\text{LM Loss} = \text{CrossEntropy}(\text{shifted logits}, \text{shifted labels})
$$

### 2. Comprehensive Reward Function
The reward used for neuroplasticity and curriculum advancement is a weighted sum of several metrics:

$$
\text{Total Reward} = w_\text{rouge} \cdot \text{ROUGE} + w_\text{coherence} \cdot \text{Coherence} + w_\text{diversity} \cdot \text{Diversity} + w_\text{perplexity} \cdot \text{Perplexity}
$$

Where:
- $\text{ROUGE}$: Mean ROUGE score between generated and reference text
- $\text{Coherence}$: Unique word ratio (proxy for repetition)
- $\text{Diversity}$: Vocabulary richness (unique words / 50, capped at 1)
- $\text{Perplexity}$: $1 - (\text{perplexity} - 1)/10$ (lower is better)
- $w_*$: Tunable weights in config

### 3. Plasticity Trace Update (Exponential Moving Average)
For each parameter $\theta$:

$$
\text{Trace}_{t+1} = \gamma \cdot \text{Trace}_t + (1 - \gamma) \cdot \nabla_\theta L
$$

Where $\gamma$ is the trace decay factor, and $\nabla_\theta L$ is the gradient.

### 4. Plasticity Modulation (Momentum + Low-Rank)
For each parameter:

$$
\text{Momentum}_{t+1} = m \cdot \text{Momentum}_t + (1 - m) \cdot \text{Trace}_{t+1}
$$

$$
\text{Modulation} = r \cdot s \cdot \text{Momentum}_{t+1} + 0.1 \cdot (U V) \quad \text{(if low-rank used)}
$$
Where $r$ is the reward signal, $s$ is the modulation strength, $U, V$ are low-rank matrices.

### 5. Plasticity Parameter Update
For each parameter $\theta$:

$$
\theta \leftarrow \theta + \eta_\text{plasticity} \cdot \text{Modulation}
$$

Where $\eta_\text{plasticity}$ is the plasticity learning rate.

---

For more details, see the code in `neuro-transformer/model1.py` and the inline documentation.
