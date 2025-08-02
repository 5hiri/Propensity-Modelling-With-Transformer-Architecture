"""
Character-level tokenizer for simple testing without external dependencies.
"""

from typing import List, Dict, Optional


class CharTokenizer:
    """A simple character-level tokenizer for testing."""
    
    def __init__(self, text_corpus: Optional[str] = None):
        """Initialize the character tokenizer.
        
        Args:
            text_corpus: Text to build vocabulary from
        """
        # Default character set if no corpus provided
        if text_corpus is None:
            chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-()[]{}'\"/\\n\\t")
        else:
            chars = sorted(list(set(text_corpus)))
        
        # Add special tokens
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.chars = special_tokens + chars
        
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Special token IDs
        self.pad_token_id = self.char_to_idx["<PAD>"]
        self.unk_token_id = self.char_to_idx["<UNK>"]
        self.bos_token_id = self.char_to_idx["<BOS>"]
        self.eos_token_id = self.char_to_idx["<EOS>"]
    
    def encode(self, text: str, max_length: Optional[int] = None, 
               padding: bool = False, truncation: bool = True) -> List[int]:
        """Encode text to token IDs."""
        # Convert characters to IDs
        ids = []
        for ch in text:
            if ch in self.char_to_idx:
                ids.append(self.char_to_idx[ch])
            else:
                ids.append(self.unk_token_id)
        
        # Handle max_length
        if max_length is not None:
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            elif padding and len(ids) < max_length:
                ids.extend([self.pad_token_id] * (max_length - len(ids)))
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        chars = []
        for token_id in token_ids:
            if token_id in self.idx_to_char:
                char = self.idx_to_char[token_id]
                if skip_special_tokens and char in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                    continue
                chars.append(char)
        return ''.join(chars)
