Neuro-Plastic Fine-Tuning for Controlled Text Generation
This repository contains a Python script for fine-tuning a GPT-2 model using a novel, neuro-inspired approach. The goal is to produce a model that generates text controlled for specific qualities like length, semantic relevance, and diversity, moving beyond standard cross-entropy loss.
The implementation combines several advanced techniques:
Neuro-Plasticity: A custom wrapper that applies Hebbian-like weight updates based on reward signals, allowing the model to "learn how to learn."
Multi-Objective Reward Function: A sophisticated loss function that guides the model towards generating efficient, semantically coherent, diverse, and contextually appropriate text.
Curriculum Learning: A structured training regimen that starts with simple tasks and gradually moves to more complex ones, stabilizing the learning process.
Mixed-Precision Training: Utilizes CUDA's capabilities for faster and more memory-efficient training.
How It Works
The core of this project is the NeuroPlasticWrapper, which augments the standard backpropagation process.
Standard Forward/Backward Pass: The model first calculates a standard language modeling loss (Cross-Entropy) and a custom reward_loss. This reward_loss is a weighted sum of penalties for undesirable outputs (e.g., too long, semantically irrelevant, not diverse).
Gradient-Based Reward Signal: The gradients from this combined loss are used for two purposes. First, they update the model weights via a standard optimizer (AdamW).
Eligibility Traces: The NeuroPlasticWrapper maintains "eligibility traces" for each weight in the network. During the backward pass, it observes which weights have high gradients, marking them as "eligible" for plastic changes. These traces accumulate over time, much like synaptic potentiation in the brain.
Plastic Update: After the optimizer step, a second, separate update is applied. The model's most "eligible" weights are modified based on the reward signal (derived from the negative gradient). This update is very small (plasticity_lr) and acts as a meta-learning mechanism, reinforcing pathways that lead to high-reward outcomes.
This dual-update system allows the model to both learn the language task directly (via AdamW) and learn to configure its own weights to become better at the task over time (via the plastic updates).
Key Components
1. Neuro-Plasticity Wrapper (NeuroPlasticWrapper)
Wraps the gpt2-medium model.
Maintains eligibility_traces for model parameters, inspired by biological synapses.
Updates these traces based on the magnitude of parameter gradients.
Applies small, reward-driven weight updates (plastic_update) to reinforce beneficial learning pathways.
2. Multi-Objective Reward Function
The total loss is a combination of the standard cross-entropy loss and a weighted reward loss:
total_loss = loss_ce + lambda_reward * reward_loss
The reward_loss is composed of four distinct components:
Efficiency Loss (alpha): Penalizes the model for generating text that deviates from a target_length.
Semantic Loss (beta_sem): Encourages the generated response to be semantically similar to the prompt, using cosine similarity between prompt and response embeddings.
Diversity Loss (gamma_div): Promotes creativity and reduces repetition by rewarding higher entropy in the model's output probability distribution.
Reference-based Loss (delta_ref): Uses BLEU and METEOR scores to penalize the model for deviating from a ground-truth response, ensuring factual and stylistic alignment.
3. Curriculum Learning (CURRICULUM)
Training proceeds through a predefined curriculum to ensure stability and effective learning. The model starts with simpler, shorter tasks and progresses to more complex, longer-form generation.
Definitions: Short, factual statements (max length 30).
Explanations: More detailed, explanatory text (max length 80).
Dialogues: Complex, conversational generation (max length 150).
Setup and Installation
1. Clone the Repository
Generated bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Use code with caution.
Bash
2. Create a Virtual Environment (Recommended)
Generated bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Use code with caution.
Bash
3. Install Dependencies
The required packages are listed in requirements.txt.
Generated bash
pip install -r requirements.txt
Use code with caution.
Bash
If you don't have a requirements.txt file, create one with the following content:
Generated code
torch
transformers
numpy
nltk
datasets
Use code with caution.
Then run the pip install command.
4. Download NLTK Data
The script will automatically download the necessary NLTK resources (wordnet, omw-1.4, punkt) on its first run.
5. Prepare the Dataset
The script is designed to load data from a local file named dolly_data.csv. This file is expected to have instruction and response columns.
You can create this file from the popular Databricks Dolly 15k dataset.
Steps to create dolly_data.csv:
a. Download the dataset (e.g., using the datasets library).
b. Run the following Python snippet to convert it to the required CSV format:
Generated python
from datasets import load_dataset
import pandas as pd

# Load the dataset from Hugging Face
dataset = load_dataset("databricks/dolly-v15k", split="train")

# Convert to a pandas DataFrame
df = dataset.to_pandas()

# Keep only the relevant columns
df = df[['instruction', 'context', 'response']]

# Save to CSV
df.to_csv('dolly_data.csv', index=False)

print("dolly_data.csv has been created successfully.")
Use code with caution.
Python
Place the resulting dolly_data.csv file in the root directory of the project. If the file is not found, the script will fall back to using synthetic data for demonstration purposes.
How to Run
To start the training process, simply run the Python script. The script will automatically detect and use a CUDA-enabled GPU if available, otherwise it will fall back to the CPU.
Generated bash
python your_script_name.py
Use code with caution.
Bash
Training progress, including loss and reward metrics, will be printed to the console.
The best-performing model checkpoint for each curriculum stage will be saved in the checkpoints/ directory (e.g., best_model_definitions.pt).
After training is complete, a final evaluation will run, generating sample text for a few test concepts.
Configuration
You can tweak the model's behavior by adjusting the hyperparameters at the top of the script:
Parameter	Description	Default
target_length	The ideal length (in tokens) for generated responses.	50
plasticity_lr	Learning rate for the neuro-plastic weight updates.	1e-6
gamma	Decay factor for eligibility traces (how quickly they "forget").	0.95
beta	A scaling factor for the plastic update.	1.0
lambda_decay	Weight decay applied during the plastic update to prevent drift.	0.01
lambda_reward	Initial weight of the reward_loss relative to the loss_ce.	0.2
alpha	Weight of the Efficiency loss component.	5.0
beta_sem	Weight of the Semantic loss component.	0.7
gamma_div	Weight of the Diversity loss component.	0.3
delta_ref	Weight of the Reference-based loss component.	0.4
Future Work
Experiment with different reward functions: Incorporate other metrics like perplexity, factuality checks, or toxicity scores.
Advanced Curriculum: Create a more dynamic curriculum that adapts based on the model's performance.
Different Models: Adapt the NeuroPlasticWrapper for other model architectures like T5, Llama, or BERT.
Hyperparameter Search: Systematically search for optimal values of the plasticity and reward loss weights.
