# Attention Mechanisms in SimpleLLM

## Purpose and Role

The attention mechanism is the core innovation that makes transformers so powerful. It allows the model to dynamically focus on different parts of the input sequence when processing each token, enabling it to capture long-range dependencies and complex relationships between words regardless of their distance in the sequence.

In SimpleLLM, we implement two key attention components:
- **MultiHeadAttention**: The fundamental attention mechanism that can attend to any position
- **CausalSelfAttention**: A specialized version for autoregressive language modeling that prevents the model from "looking ahead" at future tokens

## Key Classes and Functions

### MultiHeadAttention
The core attention mechanism that implements the "Attention is All You Need" paper's multi-head attention.

**Key responsibilities:**
- Transforms input into Query (Q), Key (K), and Value (V) representations
- Computes attention scores between all pairs of positions
- Applies attention weights to values to create context-aware representations
- Combines multiple attention "heads" for richer representations

### CausalSelfAttention
A wrapper around MultiHeadAttention that adds causal masking for language modeling.

**Key responsibilities:**
- Ensures autoregressive property (can't attend to future tokens)
- Combines causal masking with optional padding masks
- Provides the self-attention mechanism where Q, K, and V all come from the same input

## Mathematical Concepts

### Scaled Dot-Product Attention

The fundamental attention operation computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Why this works:**
1. **QK^T**: Computes similarity scores between queries and keys
2. **√d_k scaling**: Prevents softmax saturation in high dimensions
3. **Softmax**: Converts scores to probability distribution (attention weights)
4. **Multiply by V**: Weighted combination of values based on attention

### Multi-Head Attention

Instead of using a single attention function, we use multiple "heads":

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Benefits:**
- Each head can focus on different types of relationships
- Parallel processing of different representation subspaces
- Richer, more nuanced attention patterns

### Causal Masking

For language modeling, we need to prevent the model from seeing future tokens:

```
mask[i,j] = 1 if j ≤ i else 0
```

This creates a lower triangular matrix that masks out future positions.

## Code Flow and Data Shapes

### MultiHeadAttention Forward Pass

```python
# Input: [batch_size, seq_len, d_model]
query, key, value = x, x, x  # Self-attention case

# 1. Linear projections
Q = W_q(query)  # [batch_size, seq_len, d_model]
K = W_k(key)    # [batch_size, seq_len, d_model]  
V = W_v(value)  # [batch_size, seq_len, d_model]

# 2. Reshape for multi-head attention
Q = Q.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
# Result: [batch_size, n_heads, seq_len, d_k]

# 3. Compute attention scores
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
# Result: [batch_size, n_heads, seq_len, seq_len]

# 4. Apply mask and softmax
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
attention_weights = softmax(scores, dim=-1)

# 5. Apply attention to values
output = attention_weights @ V
# Result: [batch_size, n_heads, seq_len, d_k]

# 6. Concatenate heads and project
output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
output = W_o(output)  # [batch_size, seq_len, d_model]
```

### CausalSelfAttention Flow

```python
# Input: [batch_size, seq_len, d_model]
seq_len = x.size(1)

# 1. Get causal mask (lower triangular)
causal_mask = self.causal_mask[:seq_len, :seq_len]
# Shape: [seq_len, seq_len]

# 2. Combine with padding mask if provided
if padding_mask is not None:
    combined_mask = causal_mask * padding_mask
else:
    combined_mask = causal_mask

# 3. Apply multi-head attention with combined mask
output, attention_weights = self.attention(x, x, x, combined_mask)
```

## Integration with Other Components

### Connection to Embeddings
- Attention operates on embedded representations from the embeddings module
- Input shape: `[batch_size, seq_len, d_model]` matches embedding output

### Connection to Transformer Blocks
- CausalSelfAttention is used within each TransformerBlock
- Output feeds into the feed-forward network
- Residual connections preserve information flow

### Connection to Generation
- During text generation, attention weights determine which previous tokens influence the next token prediction
- Causal masking ensures the autoregressive property is maintained

## Implementation Details and Design Choices

### Pre-computed Causal Mask
```python
causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
self.register_buffer('causal_mask', causal_mask)
```

**Why this approach:**
- Efficiency: Mask is computed once and reused
- Memory: Stored as a buffer, moves with model to GPU/CPU
- Flexibility: Can handle variable sequence lengths up to maximum

### Scaling Factor
```python
self.scale = math.sqrt(self.d_k)
```

**Purpose:**
- Prevents attention scores from becoming too large
- Maintains stable gradients during training
- Based on theoretical analysis in the original paper

### Dropout in Attention
```python
attention_weights = self.dropout(attention_weights)
```

**Benefits:**
- Regularization: Prevents overfitting to specific attention patterns
- Robustness: Forces model to use multiple attention paths
- Training stability: Reduces variance in attention weights

### Mask Handling
The implementation carefully handles different mask dimensions:
- 2D masks: `[seq_len, seq_len]` - basic causal or padding masks
- 3D masks: `[batch_size, seq_len, seq_len]` - batch-specific masks
- 4D masks: `[batch_size, n_heads, seq_len, seq_len]` - head-specific masks

This flexibility allows for complex masking scenarios while maintaining efficiency.

### Memory Efficiency
- Uses `torch.matmul` for efficient batch matrix multiplication
- Reshaping operations use `contiguous()` to ensure memory layout
- Buffer registration keeps masks on the same device as model parameters

The attention mechanism is the heart of the transformer architecture, enabling the model to build rich, context-aware representations that capture both local and global dependencies in the input sequence.
