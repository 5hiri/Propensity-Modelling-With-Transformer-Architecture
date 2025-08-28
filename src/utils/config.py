"""
Configuration management for the Simple LLM project.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    
    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer size
    max_seq_len: int = 512   # Maximum sequence length
    d_model: int = 384       # Embedding dimension
    n_layers: int = 6        # Number of transformer blocks
    n_heads: int = 6         # Number of attention heads
    d_ff: int = 1536         # Feed-forward dimension (usually 4 * d_model)
    dropout: float = 0.1     # Dropout probability
    num_classes: int = 2     # Number of classes for classification tasks
    
    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 10
    warmup_steps: int = 1000
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Generation
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 100
    
    # Paths
    data_path: str = "data"
    model_save_path: str = "models"
    log_dir: str = "logs"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_ff > 0, "Feed-forward dimension must be positive"
        assert self.n_layers > 0, "Number of layers must be positive"

# Default configuration instance
default_config = ModelConfig()


def get_small_config() -> ModelConfig:
    """Get a small model configuration for testing."""
    return ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=256,
        batch_size=4,
        max_epochs=5
    )


def get_medium_config() -> ModelConfig:
    """Get a medium model configuration."""
    return ModelConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        max_seq_len=1024,
        batch_size=6
    )


def get_large_config() -> ModelConfig:
    """Get a large model configuration."""
    return ModelConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=1024,
        batch_size=4
    )

def get_small_classifier_config() -> ModelConfig:
    """Get a small model configuration for text classification."""
    return ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=256,
        batch_size=4,
        max_epochs=5
    )

def get_medium_classifier_config() -> ModelConfig:
    """Get a medium model configuration for text classification."""
    return ModelConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        max_seq_len=4096,
        batch_size=4,
        max_epochs=5
    )

def get_large_classifier_config() -> ModelConfig:
    """Get a large model configuration for text classification."""
    return ModelConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=8192,
        batch_size=4,
        max_epochs=5
    )