# TinyGPT Studio

TinyGPT Studio is an end-to-end project for training, evaluating, and
deploying a small character-level GPT-style language model from scratch.

The project demonstrates practical machine-learning engineering skills:
data preparation, transformer implementation, training-loop design,
experiment tracking, checkpointing, evaluation, and interactive deployment.

## Live demo
Try the trained model here:
https://tinygpt-studio-adarsh.streamlit.app/

## What I built
- A character-level GPT-style transformer implemented in PyTorch
- A custom training loop with validation and checkpointing
- Learning-rate warmup and cosine decay
- Gradient clipping and early stopping
- W&B experiment tracking
- Train, validation, and test evaluation
- Perplexity-based model evaluation
- Interactive Streamlit text-generation app
- Resumable training checkpoints
- Best-checkpoint selection based on validation loss

## Architecture
The model uses:
- Character-level tokenization
- Token and positional embeddings
- Causal multi-head self-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- A linear language-model head

## Training setup
- Optimizer: AdamW
- Warmup iterations: 300
- Maximum iterations: 12,000
- Learning rate: 3e-4
- Minimum learning rate: 3e-6
- Batch size: 32
- Context length: 128
- Gradient clipping: 1.0

## Training metrics
Training metrics, validation loss, perplexity, learning-rate schedule, and
generated samples are tracked with Weights & Biases.

View the training run here:
https://forge.coreweave.com/wandb/adarsh_d_mech-none/tinygpt-from-scratch/runs/0d6qfyb3


## How to run
```bash
pip install -r requirements.txt
python train_tinygpt.py
Add wandb api to streamlit secrets
streamlit run app.py
```

## Project structure
```text
.
├── app.py
├── model.py
├── train_tinygpt.py
├── requirements.txt
├── README.md
├── data/
└── output/
```

## Future work
- Add BPE tokenization
- Train on a larger dataset
- Add top-p sampling
- Add beam search
- Benchmark multiple model sizes
- Add automated tests
- Deploy with Docker

- Add model quantization
