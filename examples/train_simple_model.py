"""
Example script for training a simple language model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model.transformer import SimpleLLM
from src.training.trainer import LMTrainer
from src.training.data_loader import prepare_data
from src.utils.config import get_small_config
from src.utils.tokenizer import SimpleTokenizer


def main():
    """Main training function."""
    print("=== Simple LLM Training Example ===\n")
    
    # Load configuration
    config = get_small_config()  # Use small config for faster training
    print(f"Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = SimpleTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Update config with actual vocab size
    config.vocab_size = tokenizer.vocab_size
    
    # Prepare data
    print("\nPreparing data...")
    train_loader, val_loader = prepare_data(
        config.data_path, 
        tokenizer, 
        config.batch_size, 
        config.max_seq_len
    )
    
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
    print(f"Model config: {model.get_config()}")
    
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
        "Machine learning is",
        "The transformer architecture",
        "Natural language processing"
    ]
    
    model.eval()
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_ids = input_ids.to(trainer.device)
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.8,
                top_k=50,
                do_sample=True
            )
        
        # Decode and print
        generated_text = tokenizer.decode(generated[0])
        print(f"Generated: '{generated_text}'")


if __name__ == "__main__":
    main()
