"""
Tokenizer utilities for the Simple LLM project.
"""

import torch
from typing import List, Union
from transformers import GPT2Tokenizer


class SimpleTokenizer:
    """A simple tokenizer wrapper around GPT-2 tokenizer."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the tokenizer.
        
        Args:
            model_name: The model name for the tokenizer (default: gpt2)
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        """Get the padding token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id
    
    def encode(self, text: Union[str, List[str]], 
               max_length: int = None, 
               padding: bool = False,
               truncation: bool = True) -> torch.Tensor:
        """Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Tensor of token IDs
        """
        if isinstance(text, str):
            text = [text]
        
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        
        return encoded["input_ids"]
    
    def decode(self, token_ids: torch.Tensor, 
               skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """Decode token IDs to text.
        
        Args:
            token_ids: Tensor of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text or list of texts
        """
        if token_ids.dim() == 1:
            # Single sequence
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            # Batch of sequences
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Create attention mask for input sequences.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        pad_token_id: Padding token ID
        
    Returns:
        Attention mask [batch_size, seq_len]
    """
    return (input_ids != pad_token_id).float()


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Create causal (lower triangular) mask for self-attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Causal mask [seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.bool()
