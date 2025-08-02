"""
Simple LLM Implementation Package

A complete implementation of a transformer-based language model from scratch.
"""

__version__ = "1.0.0"
__author__ = "Simple LLM Project"

# Make key components easily accessible
from .model.transformer import SimpleLLM
from .model.attention import MultiHeadAttention, CausalSelfAttention
from .model.embeddings import InputEmbedding, PositionalEncoding
from .training.trainer import LMTrainer
from .generation.generator import TextGenerator
from .utils.config import ModelConfig, default_config
from .utils.tokenizer import SimpleTokenizer

__all__ = [
    "SimpleLLM",
    "MultiHeadAttention", 
    "CausalSelfAttention",
    "InputEmbedding",
    "PositionalEncoding",
    "LMTrainer",
    "TextGenerator", 
    "ModelConfig",
    "default_config",
    "SimpleTokenizer"
]
