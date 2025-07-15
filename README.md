# Neuro-Plastic Fine-Tuning for Controlled Text Generation

This repository contains a Python script for fine-tuning a GPT-2 model using a novel, neuro-inspired approach. The goal is to produce a model that generates text controlled for specific qualities like length, semantic relevance, and diversity, moving beyond standard cross-entropy loss.

The implementation combines several advanced techniques:
- **Neuro-Plasticity:** A custom wrapper that applies Hebbian-like weight updates based on reward signals, allowing the model to "learn how to learn."
- **Multi-Objective Reward Function:** A sophisticated loss function that guides the model towards generating efficient, semantically coherent, diverse, and contextually appropriate text.
- **Curriculum Learning:** A structured training regimen that starts with simple tasks and gradually moves to more complex ones, stabilizing the learning process.
- **Mixed-Precision Training:** Utilizes CUDA's capabilities for faster and more memory-efficient training.

## How It Works

The core of this project is the `NeuroPlasticWrapper`, which augments the standard backpropagation process.

1.  **Standard Forward/Backward Pass:** The model first calculates a standard language modeling loss (Cross-Entropy) and a custom `reward_loss`. This `reward_loss` is a weighted sum of penalties for undesirable outputs (e.g., too long, semantically irrelevant, not diverse).
2.  **Gradient-Based Reward Signal:** The gradients from this combined loss are used for two purposes. First, they update the model weights via a standard optimizer (`AdamW`).
3.  **Eligibility Traces:** The `NeuroPlasticWrapper` maintains "eligibility traces" for each weight in the network. During the backward pass, it observes which weights have high gradients, marking them as "eligible" for plastic changes. These traces accumulate over time, much like synaptic potentiation in the brain.
4.  **Plastic Update:** After the optimizer step, a second, separate update is applied. The model's most "eligible" weights are modified based on the reward signal (derived from the negative gradient). This update is very small (`plasticity_lr`) and acts as a meta-learning mechanism, reinforcing pathways that lead to high-reward outcomes.

This dual-update system allows the model to both learn the language task directly (via `AdamW`) and learn to configure its own weights to become better at the task over time (via the plastic updates).

## Key Components

### 1. Neuro-Plasticity Wrapper (`NeuroPlasticWrapper`)
- Wraps the `gpt2-medium` model.
- Maintains `eligibility_traces` for model parameters, inspired by biological synapses.
- Updates these traces based on the magnitude of parameter gradients.
- Applies small, reward-driven weight updates (`plastic_update`) to reinforce beneficial learning pathways.

### 2. Multi-Objective Reward Function
The total loss is a combination of the standard cross-entropy loss and a weighted reward loss:
`total_loss = loss_ce + lambda_reward * reward_loss`

The `reward_loss` is composed of four distinct components:
- **Efficiency Loss (`alpha`):** Penalizes the model for generating text that deviates from a `target_length`.
- **Semantic Loss (`beta_sem`):** Encourages the generated response to be semantically similar to the prompt, using cosine similarity between prompt and response embeddings.
- **Diversity Loss (`gamma_div`):** Promotes creativity and reduces repetition by rewarding higher entropy in the model's output probability distribution.
- **Reference-based Loss (`delta_ref`):** Uses **BLEU** and **METEOR** scores to penalize the model for deviating from a ground-truth response, ensuring factual and stylistic alignment.

### 3. Curriculum Learning (`CURRICULUM`)
Training proceeds through a predefined curriculum to ensure stability and effective learning. The model starts with simpler, shorter tasks and progresses to more complex, longer-form generation.
1.  **Definitions:** Short, factual statements (max length 30).
2.  **Explanations:** More detailed, explanatory text (max length 80).
3.  **Dialogues:** Complex, conversational generation (max length 150).

---
### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv



## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
