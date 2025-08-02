"""
Simple training example using character-level tokenization.
"""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.transformer import SimpleLLM
from src.training.trainer import LMTrainer
from src.training.data_loader import prepare_data
from src.utils.config import get_small_config
from src.utils.char_tokenizer import CharTokenizer
from src.generation.generator import TextGenerator


def main():
    """Main training function with character tokenizer."""
    print("=== Simple LLM Character-Level Training ===\n")
    
    # Sample training text
    training_text = """
    Machine learning is a powerful technology. Deep learning uses neural networks.
    Transformers are the backbone of modern language models. Attention mechanisms help models focus.
    Natural language processing enables computers to understand text. 
    Large language models can generate coherent text by predicting the next character.
    Training requires lots of data and computational power. The future of AI is very exciting.
    """
    
    # Load configuration
    config = get_small_config()
    config.max_epochs = 20  # Quick training
    config.batch_size = 4
    config.max_seq_len = 64
    print(f"Configuration: {config.d_model}d model, {config.n_layers} layers, {config.max_epochs} epochs")
    
    # Initialize tokenizer
    print("\nInitializing character tokenizer...")
    tokenizer = CharTokenizer(training_text)
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create sample data files and use direct text
    print("\nPreparing data...")
    longer_text = training_text * 10  # Make longer for better training
    train_texts = [longer_text]
    
    # Create data loaders directly
    from src.training.data_loader import create_data_loader
    train_loader = create_data_loader(train_texts, tokenizer, config.batch_size, config.max_seq_len, shuffle=True)
    val_loader = create_data_loader([training_text], tokenizer, config.batch_size, config.max_seq_len, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = SimpleLLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = LMTrainer(model, config)
    
    # Start training
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")
    
    # Test generation
    print("\n=== Testing Text Generation ===")
    
    # Load the best model
    best_model_path = os.path.join(config.model_save_path, "best_model.pt")
    if os.path.exists(best_model_path):
        print("Loading best model for generation test...")
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate some sample text
    test_prompts = [
        "Machine learning",
        "Deep learning",
        "Transformers",
        "Natural language"
    ]
    
    generator = TextGenerator(model, tokenizer)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generator.generate_text(
            prompt,
            max_new_tokens=30,
            temperature=0.8,
            top_k=10
        )
        print(f"Generated: '{generated}'")

    print("\n🎉 Character-level training completed successfully!")


if __name__ == "__main__":
    main()
