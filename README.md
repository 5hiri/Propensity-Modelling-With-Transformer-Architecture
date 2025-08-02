# Simple LLM from Scratch

A simple implementation of a modern Large Language Model (LLM) using PyTorch, built from scratch for educational purposes.

## Features

- **Transformer Architecture**: Complete implementation of the transformer model with multi-head attention
- **Modern Components**: Layer normalization, residual connections, positional encoding
- **Training Pipeline**: Full training loop with loss tracking and checkpointing
- **Text Generation**: Inference capabilities with different sampling strategies
- **Configurable**: Easy-to-modify hyperparameters and model architecture

## Project Structure

```
├── src/
│   ├── model/
│   │   ├── transformer.py      # Main transformer implementation
│   │   ├── attention.py        # Multi-head attention mechanism
│   │   └── embeddings.py       # Token and positional embeddings
│   ├── training/
│   │   ├── trainer.py          # Training loop and utilities
│   │   └── data_loader.py      # Data loading and preprocessing
│   ├── generation/
│   │   └── generator.py        # Text generation utilities
│   └── utils/
│       ├── config.py           # Configuration management
│       └── tokenizer.py        # Tokenization utilities
├── examples/
│   ├── train_simple_model.py   # Training example
│   └── generate_text.py        # Text generation example
├── notebooks/
│   └── exploration.ipynb       # Jupyter notebook for exploration
└── data/                       # Training data directory
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a Simple Model**:
   ```bash
   python examples/train_simple_model.py
   ```

3. **Generate Text**:
   ```bash
   python examples/generate_text.py
   ```

## Model Architecture

This implementation includes:

- **Multi-Head Self-Attention**: The core mechanism that allows the model to attend to different parts of the input
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Layer Normalization**: For training stability
- **Residual Connections**: To help with gradient flow
- **Positional Encoding**: To give the model information about token positions

## Learning Resources

This project is designed to be educational. Key concepts implemented:

1. **Attention Mechanism**: Understanding how transformers "pay attention" to different parts of the input
2. **Transformer Blocks**: The fundamental building blocks of modern LLMs
3. **Training Loop**: How to train a language model from scratch
4. **Text Generation**: Different strategies for generating text (greedy, sampling, top-k, etc.)

## Configuration

Modify `src/utils/config.py` to experiment with different:
- Model sizes (embedding dimensions, number of layers, attention heads)
- Training hyperparameters (learning rate, batch size, etc.)
- Generation parameters (temperature, top-k, top-p)

## Next Steps

Once you understand this basic implementation, you can:
- Scale up the model size
- Implement more advanced techniques (RoPE, RMSNorm, etc.)
- Add more sophisticated training techniques
- Experiment with different datasets
