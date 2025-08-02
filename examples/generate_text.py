"""
Example script for text generation with a trained model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model.transformer import SimpleLLM
from src.generation.generator import TextGenerator
from src.utils.config import get_small_config
from src.utils.tokenizer import SimpleTokenizer


def load_trained_model(model_path: str, config):
    """Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        config: Model configuration
        
    Returns:
        Loaded model
    """
    # Initialize model
    model = SimpleLLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    # Load checkpoint
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print(f"No trained model found at {model_path}")
        print("Please train a model first using train_simple_model.py")
        return None
    
    return model


def demo_generation_strategies(generator: TextGenerator, prompt: str):
    """Demonstrate different text generation strategies.
    
    Args:
        generator: Text generator instance
        prompt: Input prompt
    """
    print(f"\n=== Generation Strategies Comparison ===")
    print(f"Prompt: '{prompt}'\n")
    
    strategies = generator.compare_generation_strategies(prompt, max_new_tokens=40)
    
    for strategy_name, generated_text in strategies.items():
        print(f"{strategy_name.upper().replace('_', ' ')}:")
        print(f"  {generated_text}")
        print()


def main():
    """Main generation function."""
    print("=== Simple LLM Text Generation ===\n")
    
    # Load configuration
    config = get_small_config()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = SimpleTokenizer()
    config.vocab_size = tokenizer.vocab_size
    
    # Load trained model
    model_path = os.path.join(config.model_save_path, "best_model.pt")
    model = load_trained_model(model_path, config)
    
    if model is None:
        return
    
    # Initialize text generator
    generator = TextGenerator(model, tokenizer)
    
    # Demo prompts
    demo_prompts = [
        "Machine learning is",
        "The future of artificial intelligence",
        "Deep learning models",
        "Natural language processing enables",
        "Transformers have revolutionized"
    ]
    
    print("\n=== Basic Text Generation ===")
    for prompt in demo_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generator.generate_text(
            prompt, 
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        print(f"Generated: {generated}")
    
    # Demonstrate different generation strategies
    demo_generation_strategies(generator, "Machine learning is")
    
    # Interactive generation
    print("\n=== Interactive Generation ===")
    print("You can now enter prompts interactively!")
    print("(This will work best with a larger, well-trained model)")
    
    try:
        generator.interactive_generation(
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    
    print("\nGeneration demo completed!")


if __name__ == "__main__":
    main()
