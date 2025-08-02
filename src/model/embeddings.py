"""
Token and positional embeddings for the transformer model.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TokenEmbedding(nn.Module):
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, d_model: int):
        """Initialize token embedding.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input token IDs [batch_size, seq_len]
            
        Returns:
            Token embeddings [batch_size, seq_len, d_model]
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
        
        # Type annotation for the buffer to fix type checking
        self.pe: torch.Tensor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding (alternative to sinusoidal)."""
    
    def __init__(self, max_seq_len: int, d_model: int):
        """Initialize learned positional embedding.
        
        Args:
            max_seq_len: Maximum sequence length
            d_model: Model dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to input.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.embedding(positions)


class InputEmbedding(nn.Module):
    """Combined token and positional embeddings."""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, 
                 dropout: float = 0.1, use_learned_pe: bool = False):
        """Initialize input embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_learned_pe: Whether to use learned positional embeddings
        """
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        if use_learned_pe:
            self.pos_embedding = LearnedPositionalEmbedding(max_seq_len, d_model)
        else:
            self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding layers.
        
        Args:
            x: Input token IDs [batch_size, seq_len]
            
        Returns:
            Embedded input [batch_size, seq_len, d_model]
        """
        # Get token embeddings
        token_emb = self.token_embedding(x)
        
        # Add positional encoding
        embedded = self.pos_embedding(token_emb)
        
        # Apply dropout
        return self.dropout(embedded)
