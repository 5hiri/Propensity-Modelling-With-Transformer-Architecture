# Transformer Architecture in SimpleLLM

## Purpose and Role

The transformer module contains the core architecture components that make up the SimpleLLM language model. It orchestrates the flow of information through multiple transformer blocks, each containing self-attention and feed-forward networks, to build increasingly sophisticated representations of the input text.

The main components work together to:
- Process embedded tokens through multiple layers of self-attention and feed-forward networks
- Build hierarchical representations from simple token meanings to complex contextual understanding
- Generate probability distributions over the vocabulary for next-token prediction
- Support both training and autoregressive text generation

## Key Classes and Functions

### FeedForward
A position-wise feed-forward network that processes each position independently.

**Key responsibilities:**
- Applies non-linear transformations to attention outputs
- Expands and contracts the representation dimensionality
- Provides the model's primary source of non-linearity and expressiveness

### TransformerBlock
A complete transformer layer combining self-attention and feed-forward processing.

**Key responsibilities:**
- Integrates causal self-attention with feed-forward processing
- Implements residual connections and layer normalization
- Forms the repeatable building block of the transformer architecture

### SimpleLLM
The complete language model that orchestrates all components.

**Key responsibilities:**
- Manages the full forward pass from tokens to logits
- Implements autoregressive text generation
- Provides model configuration and parameter counting utilities
- Handles weight initialization and model setup

## Mathematical Concepts

### Feed-Forward Network

The feed-forward network applies a two-layer MLP to each position:

```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

**Key properties:**
- **Position-wise**: Same transformation applied to each position independently
- **Expansion**: Typically d_ff = 4 × d_model for increased expressiveness
- **Non-linearity**: ReLU activation enables complex pattern learning

### Residual Connections and Layer Normalization

Each sub-layer uses the pattern:
```
output = x + Sublayer(LayerNorm(x))
```

**Benefits of Pre-Norm (used in SimpleLLM):**
- More stable training than post-norm
- Better gradient flow through deep networks
- Reduced risk of exploding/vanishing gradients

### Transformer Block Computation

```
# Self-attention sub-layer
attn_input = LayerNorm(x)
attn_output, _ = SelfAttention(attn_input)
x = x + Dropout(attn_output)

# Feed-forward sub-layer  
ff_input = LayerNorm(x)
ff_output = FeedForward(ff_input)
x = x + Dropout(ff_output)
```

### Language Model Head

The final projection converts hidden states to vocabulary logits:
```
logits = hidden_states @ W_vocab
```

Where W_vocab can be tied to the input embedding weights for parameter efficiency.

## Code Flow and Data Shapes

### Complete Forward Pass

```python
# Input: Token IDs [batch_size, seq_len]
input_ids = torch.tensor([[1, 15, 234, 89, 2]])

# 1. Input embedding
x = self.embedding(input_ids)
# Shape: [batch_size, seq_len, d_model]

# 2. Pass through transformer blocks
for block in self.transformer_blocks:
    x = block(x, attention_mask)
    # Shape maintained: [batch_size, seq_len, d_model]

# 3. Final layer normalization
x = self.ln_f(x)
# Shape: [batch_size, seq_len, d_model]

# 4. Project to vocabulary
logits = self.lm_head(x)
# Shape: [batch_size, seq_len, vocab_size]
```

### TransformerBlock Flow

```python
def forward(self, x, mask=None):
    # Input: [batch_size, seq_len, d_model]
    
    # Self-attention with pre-norm and residual
    attn_input = self.ln1(x)  # [batch_size, seq_len, d_model]
    attn_output, _ = self.self_attention(attn_input, mask)
    x = x + self.dropout(attn_output)  # Residual connection
    
    # Feed-forward with pre-norm and residual
    ff_input = self.ln2(x)  # [batch_size, seq_len, d_model]
    ff_output = self.feed_forward(ff_input)
    x = x + self.dropout(ff_output)  # Residual connection
    
    return x  # [batch_size, seq_len, d_model]
```

### Text Generation Flow

```python
def generate(self, input_ids, max_new_tokens=50):
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # 1. Forward pass on current sequence
        logits = self.forward(generated)["logits"]
        
        # 2. Get logits for last position
        next_token_logits = logits[:, -1, :] / temperature
        
        # 3. Apply sampling strategies (top-k, top-p)
        if top_k:
            # Keep only top-k logits
            top_k_logits, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < top_k_logits[:, -1:]] = -float('inf')
        
        # 4. Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 5. Append to sequence
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

## Integration with Other Components

### Embedding Integration
- SimpleLLM starts with InputEmbedding to convert tokens to vectors
- Embedding dimension (d_model) propagates through all transformer blocks
- Positional encoding is added at the input layer only

### Attention Integration
- Each TransformerBlock contains a CausalSelfAttention layer
- Attention masks flow through the entire model
- Attention patterns become more sophisticated in deeper layers

### Output Integration
- Final layer normalization prepares representations for output projection
- Language model head converts hidden states to vocabulary probabilities
- Logits can be used for both training (with cross-entropy loss) and generation

## Implementation Details and Design Choices

### Pre-Norm vs Post-Norm Architecture

SimpleLLM uses **pre-norm** (LayerNorm before sub-layers):

```python
# Pre-norm (used in SimpleLLM)
x = x + sublayer(layer_norm(x))

# Post-norm (original transformer)
x = layer_norm(x + sublayer(x))
```

**Advantages of pre-norm:**
- More stable training dynamics
- Better gradient flow in deep networks
- Reduced need for learning rate warmup

### Weight Initialization Strategy

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
```

**Rationale:**
- Small standard deviation (0.02) prevents activation explosion
- LayerNorm initialized to identity transformation
- Consistent with GPT-style models

### Feed-Forward Dimension Choice

```python
d_ff = 4 * d_model  # Common choice
```

**Why 4x expansion:**
- Provides sufficient expressiveness for complex patterns
- Balances model capacity with computational efficiency
- Empirically validated across many transformer variants

### Dropout Placement

```python
# Applied after attention and feed-forward outputs
x = x + self.dropout(attn_output)
x = x + self.dropout(ff_output)
```

**Strategic placement:**
- Regularizes the residual connections
- Prevents overfitting to specific layer combinations
- Maintains information flow through skip connections

### Generation Sampling Strategies

**Top-k Sampling:**
- Keeps only the k most likely tokens
- Prevents sampling from very unlikely tokens
- Balances diversity and quality

**Top-p (Nucleus) Sampling:**
- Keeps tokens until cumulative probability exceeds p
- Adaptive vocabulary size based on confidence
- More sophisticated than fixed top-k

**Temperature Scaling:**
- Controls randomness in sampling
- Higher temperature = more diverse output
- Lower temperature = more focused output

### Memory Management During Generation

```python
# Truncate sequence if it exceeds max length
if current_len >= self.max_seq_len:
    input_seq = generated[:, -self.max_seq_len:]
```

**Benefits:**
- Prevents out-of-memory errors during long generation
- Maintains causal attention properties
- Enables generation of arbitrarily long sequences

### Model Configuration and Introspection

```python
def get_num_params(self):
    return sum(p.numel() for p in self.parameters())

def get_config(self):
    return {
        "vocab_size": self.vocab_size,
        "d_model": self.d_model,
        "n_layers": len(self.transformer_blocks),
        "max_seq_len": self.max_seq_len,
    }
```

**Utility:**
- Enables model size analysis
- Supports model serialization and loading
- Facilitates architecture comparisons

The transformer architecture in SimpleLLM demonstrates how relatively simple components (attention, feed-forward, normalization) can be combined to create a powerful language model capable of understanding and generating human-like text through the emergent properties of deep, self-attentive processing.
