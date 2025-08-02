"""
Data loading and preprocessing utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any
import os
import json
from pathlib import Path


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """Initialize the dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(
                text, 
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            # Handle different tokenizer return types
            if hasattr(tokens, 'size'):  # PyTorch tensor (GPT-2 tokenizer)
                if tokens.size(1) > 1:  # Only keep non-empty sequences
                    self.tokenized_texts.append(tokens.squeeze(0))
            elif isinstance(tokens, list):  # List of tokens (char tokenizer)
                if len(tokens) > 1:  # Only keep non-empty sequences
                    self.tokenized_texts.append(torch.tensor(tokens, dtype=torch.long))
            else:  # Convert to tensor if needed
                tokens_tensor = torch.tensor(tokens, dtype=torch.long)
                if len(tokens_tensor) > 1:
                    self.tokenized_texts.append(tokens_tensor)
    
    def __len__(self) -> int:
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input_ids and labels
        """
        tokens = self.tokenized_texts[idx]
        
        # For language modeling, input and target are the same sequence
        # but shifted by one position
        if len(tokens) <= 1:
            # Handle edge case of very short sequences
            input_ids = tokens
            labels = tokens
        else:
            input_ids = tokens[:-1]  # All tokens except the last
            labels = tokens[1:]      # All tokens except the first
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }


class CollateFunction:
    """Collate function for batching sequences of different lengths."""
    
    def __init__(self, pad_token_id: int = 0):
        """Initialize collate function.
        
        Args:
            pad_token_id: Token ID used for padding
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Batched and padded tensors
        """
        # Extract input_ids and labels
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100  # -100 is ignored in loss calculation
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).float()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_text_data(data_path: str, file_pattern: str = "*.txt") -> List[str]:
    """Load text data from files.
    
    Args:
        data_path: Path to the data directory
        file_pattern: Pattern to match files
        
    Returns:
        List of text strings
    """
    data_dir = Path(data_path)
    texts = []
    
    if not data_dir.exists():
        print(f"Data directory {data_path} does not exist.")
        return texts
    
    # Find all matching files
    text_files = list(data_dir.glob(file_pattern))
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty content
                    texts.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return texts


def create_sample_data(save_path: str = "data") -> None:
    """Create sample training data for demonstration.
    
    Args:
        save_path: Path to save the sample data
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Sample texts for training
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic sentence used for typing practice.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns in data.",
        "Transformers have revolutionized natural language processing and computer vision tasks.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
        "Large language models are trained on vast amounts of text data to learn language patterns.",
        "The transformer architecture consists of encoder and decoder layers with self-attention mechanisms.",
        "Positional encoding helps transformers understand the order of tokens in a sequence.",
        "Multi-head attention allows models to attend to different types of relationships simultaneously.",
        "Training large models requires significant computational resources and careful optimization.",
        "Language models can generate coherent text by predicting the next token in a sequence.",
        "Fine-tuning pre-trained models is an effective approach for specific downstream tasks.",
        "The attention mechanism computes weighted averages of input representations.",
        "Gradient descent optimization iteratively updates model parameters to minimize loss.",
        "Overfitting occurs when a model memorizes training data rather than learning generalizable patterns.",
        "Regularization techniques help prevent overfitting and improve model generalization.",
        "Batch normalization and layer normalization stabilize training of deep neural networks.",
        "The learning rate determines how large steps the optimizer takes during training.",
        "Early stopping prevents overfitting by monitoring validation performance during training."
    ]
    
    # Save each text as a separate file
    for i, text in enumerate(sample_texts):
        file_path = os.path.join(save_path, f"sample_{i:02d}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    print(f"Created {len(sample_texts)} sample text files in {save_path}/")


def create_data_loader(texts: List[str], tokenizer, batch_size: int = 8,
                      max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """Create a data loader for training.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    dataset = TextDataset(texts, tokenizer, max_length)
    collate_fn = CollateFunction(tokenizer.pad_token_id)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )


def prepare_data(data_path: str, tokenizer, batch_size: int = 8,
                max_length: int = 512, train_split: float = 0.8) -> tuple:
    """Prepare training and validation data loaders.
    
    Args:
        data_path: Path to the data directory
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load text data
    texts = load_text_data(data_path)
    
    if not texts:
        print("No text data found. Creating sample data...")
        create_sample_data(data_path)
        texts = load_text_data(data_path)
    
    # Split into train and validation
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"Loaded {len(train_texts)} training and {len(val_texts)} validation texts")
    
    # Create data loaders
    train_loader = create_data_loader(train_texts, tokenizer, batch_size, max_length, shuffle=True)
    val_loader = create_data_loader(val_texts, tokenizer, batch_size, max_length, shuffle=False)
    
    return train_loader, val_loader
