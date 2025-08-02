# SimpleLLM Architecture Documentation

This documentation provides a comprehensive guide to understanding the transformer architecture implemented in SimpleLLM. Each module is explained with both conceptual understanding and implementation details.

## Overview

SimpleLLM is a clean, educational implementation of a transformer-based language model. The architecture follows the "Attention is All You Need" paper with modern improvements like pre-norm layers and causal attention for autoregressive generation.

### Key Components

The model consists of three main modules:

1. **[Embeddings](embeddings.md)** - Convert tokens to vectors and add positional information
2. **[Attention](attention.md)** - Multi-head self-attention mechanisms with causal masking
3. **[Transformer](transformer.md)** - Complete model architecture with transformer blocks and generation

## Architecture Flow

```
Input Tokens → Embeddings → Transformer Blocks → Output Logits
     ↓              ↓              ↓                ↓
   [1,15,234]   [vectors +    [attention +      [vocab
                positions]   feed-forward]   probabilities]
```

### Data Flow Through the Model

1. **Token IDs** `[batch_size, seq_len]` enter the model
2. **Embedding Layer** converts to dense vectors `[batch_size, seq_len, d_model]`
3. **Positional Encoding** adds position information
4. **Transformer Blocks** (repeated N times):
   - Causal self-attention with residual connections
   - Feed-forward network with residual connections
   - Layer normalization (pre-norm style)
5. **Final Layer Norm** prepares for output projection
6. **Language Model Head** projects to vocabulary `[batch_size, seq_len, vocab_size]`

## Reading Guide

### For Beginners
Start with the conceptual sections of each document:
1. [Embeddings - Purpose and Role](embeddings.md#purpose-and-role)
2. [Attention - Mathematical Concepts](attention.md#mathematical-concepts)
3. [Transformer - Integration](transformer.md#integration-with-other-components)

### For Implementers
Focus on the code flow and implementation details:
1. [Attention - Code Flow](attention.md#code-flow-and-data-shapes)
2. [Embeddings - Implementation Details](embeddings.md#implementation-details-and-design-choices)
3. [Transformer - Design Choices](transformer.md#implementation-details-and-design-choices)

### For Researchers
Examine the mathematical foundations and design rationale:
1. [Attention - Mathematical Concepts](attention.md#mathematical-concepts)
2. [Embeddings - Mathematical Concepts](embeddings.md#mathematical-concepts)
3. [Transformer - Mathematical Concepts](transformer.md#mathematical-concepts)

## Key Design Decisions

### Pre-Norm Architecture
SimpleLLM uses pre-normalization (LayerNorm before sub-layers) rather than post-norm:
- More stable training dynamics
- Better gradient flow in deep networks
- Reduced sensitivity to learning rate

### Causal Self-Attention
The model uses causal masking for autoregressive generation:
- Prevents information leakage from future tokens
- Enables parallel training while maintaining autoregressive property
- Essential for language modeling tasks

### Sinusoidal Positional Encoding
Default positional encoding uses fixed sinusoidal patterns:
- Deterministic and parameter-free
- Can extrapolate to longer sequences than seen during training
- Provides unique position signatures through frequency combinations

### Multi-Head Attention
Attention is split into multiple heads:
- Each head can focus on different types of relationships
- Parallel processing of different representation subspaces
- Richer attention patterns than single-head attention

## Model Configuration

Typical SimpleLLM configuration:
```python
model = SimpleLLM(
    vocab_size=50257,      # GPT-2 vocabulary size
    d_model=768,           # Hidden dimension
    n_layers=12,           # Number of transformer blocks
    n_heads=12,            # Number of attention heads
    d_ff=3072,             # Feed-forward dimension (4 * d_model)
    max_seq_len=1024,      # Maximum sequence length
    dropout=0.1,           # Dropout rate
    use_learned_pe=False   # Use sinusoidal positional encoding
)
```

## Performance Characteristics

### Memory Usage
- Model parameters: ~117M for the configuration above
- Attention memory: O(seq_len²) per layer
- Activation memory: O(seq_len × d_model × n_layers)

### Computational Complexity
- Training: O(seq_len² × d_model × n_layers) per forward pass
- Generation: O(seq_len × d_model × n_layers) per token
- Attention dominates for long sequences

## Extensions and Modifications

The modular design makes it easy to experiment with:

### Alternative Attention Mechanisms
- Replace `CausalSelfAttention` with other attention variants
- Add cross-attention for encoder-decoder architectures
- Implement sparse attention patterns for longer sequences

### Different Positional Encodings
- Switch between sinusoidal and learned positional embeddings
- Implement relative positional encoding
- Add rotary positional embedding (RoPE)

### Architecture Variations
- Modify feed-forward networks (e.g., use SwiGLU activation)
- Experiment with different normalization schemes
- Add additional regularization techniques

## Common Patterns

### Residual Connections
```python
x = x + sublayer(layer_norm(x))
```
Essential for training deep networks and maintaining gradient flow.

### Attention Masking
```python
scores = scores.masked_fill(mask == 0, -1e9)
```
Prevents attention to invalid positions (padding or future tokens).

### Shape Transformations
```python
x = x.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
```
Reshaping for multi-head attention while maintaining batch processing.

This documentation provides both the theoretical foundation and practical implementation details needed to understand, modify, and extend the SimpleLLM architecture. Each module builds upon the others to create a complete, functional language model suitable for both learning and research.
