"""
Multi-head attention mechanism for the transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]  
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len] or [seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections and reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.w_o(output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor [batch_size, n_heads, seq_len, d_k]
            K: Key tensor [batch_size, n_heads, seq_len, d_k]
            V: Value tensor [batch_size, n_heads, seq_len, d_k]
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match scores dimensions if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            
            # Apply mask (set masked positions to large negative value)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class CausalSelfAttention(nn.Module):
    """Causal self-attention for autoregressive language modeling."""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        """Initialize causal self-attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        # Create the underlying multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Register causal mask as buffer (lower triangular matrix)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask)
        
        # Type annotation for the buffer to fix type checking
        self.causal_mask: torch.Tensor
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through causal self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Additional attention mask (e.g., padding mask)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        seq_len = x.size(1)
        
        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # Combine with additional mask if provided
        if mask is not None:
            # Ensure mask has correct shape
            if mask.dim() == 2:  # [batch_size, seq_len] -> [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine masks (both must be 1 for attention)
            combined_mask = causal_mask.unsqueeze(0) * mask
        else:
            combined_mask = causal_mask
        
        # Self-attention (query, key, value are all the same)
        return self.attention.forward(x, x, x, combined_mask)
