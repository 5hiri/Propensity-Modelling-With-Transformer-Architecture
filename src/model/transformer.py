"""
Main transformer model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .embeddings import InputEmbedding
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Apply first linear layer with ReLU activation
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        
        # Apply second linear layer
        x = self.linear2(x)
        
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 max_seq_len: int, dropout: float = 0.1):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        # Self-attention layer
        self.self_attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm style)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection (pre-norm)
        attn_input = self.ln1(x)
        attn_output, _ = self.self_attention(attn_input, mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection (pre-norm)
        ff_input = self.ln2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)
        
        return x


class SimpleLLM(nn.Module):
    """Simple Language Model based on transformer architecture."""
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, 
                 n_heads: int, d_ff: int, max_seq_len: int, 
                 dropout: float = 0.1, use_learned_pe: bool = False):
        """Initialize the language model.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_learned_pe: Whether to use learned positional embeddings
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Input embedding layer
        self.embedding = InputEmbedding(
            vocab_size, d_model, max_seq_len, dropout, use_learned_pe
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        batch_size, seq_len = input_ids.shape
        
        # Check sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        # Get embeddings
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Apply final layer normalization
        x = self.ln_f(x)
        
        # Project to vocabulary size
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        if return_dict:
            return {"logits": logits}
        else:
            return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: Optional[int] = None, 
                 top_p: Optional[float] = None, do_sample: bool = True) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get current sequence length
                current_len = generated.size(1)
                
                # Truncate if exceeding max length
                if current_len >= self.max_seq_len:
                    input_seq = generated[:, -self.max_seq_len:]
                else:
                    input_seq = generated
                
                # Forward pass
                outputs = self.forward(input_seq, return_dict=True)
                logits = outputs["logits"]
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted indices to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_num_params(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": len(self.transformer_blocks),
            "max_seq_len": self.max_seq_len,
        }
