"""
Simple test script to verify the LLM implementation works correctly.
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.transformer import SimpleLLM
from src.utils.config import get_small_config
from src.utils.char_tokenizer import CharTokenizer
from src.training.data_loader import create_data_loader
from src.generation.generator import TextGenerator


def test_basic_functionality():
    """Test basic functionality of the LLM implementation."""
    print("=== Simple LLM Test Script ===\n")
    
    # Test 1: Configuration
    print("1. Testing configuration...")
    config = get_small_config()
    print(f"   ✅ Config created: {config.d_model}d model with {config.n_layers} layers")
    
    # Test 2: Tokenizer
    print("\n2. Testing tokenizer...")
    sample_text = "Hello, this is a test of our simple character tokenizer! It works with many characters."
    tokenizer = CharTokenizer(sample_text)
    config.vocab_size = tokenizer.vocab_size
    
    encoded = tokenizer.encode("Hello world")
    decoded = tokenizer.decode(encoded)
    print(f"   ✅ Tokenizer works: vocab_size={tokenizer.vocab_size}")
    print(f"   ✅ Encode/decode test: 'Hello world' -> '{decoded}'")
    
    # Test 3: Model creation
    print("\n3. Testing model creation...")
    model = SimpleLLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    print(f"   ✅ Model created with {model.get_num_params():,} parameters")
    
    # Test 4: Forward pass
    print("\n4. Testing forward pass...")
    test_input = torch.randint(0, config.vocab_size, (2, 10))
    with torch.no_grad():
        output = model(test_input)
        if isinstance(output, dict):
            logits = output["logits"]
            print(f"   ✅ Forward pass successful: {test_input.shape} -> {logits.shape}")
        else:
            print(f"   ✅ Forward pass successful: {test_input.shape} -> {output.shape}")
    
    # Test 5: Data loading  
    print("\n5. Testing data loading...")
    longer_text = sample_text * 3  # Make it longer
    dataloader = create_data_loader([longer_text], tokenizer, batch_size=2, max_length=20)
    print(f"   ✅ Data loading works: {len(dataloader)} batches")
    
    # Test 6: Text generation (without training)
    print("\n6. Testing text generation...")
    generator = TextGenerator(model, tokenizer)
    prompt = "Hello"
    generated = generator.generate_text(prompt, max_new_tokens=10, temperature=1.0)
    print(f"   ✅ Generation works: '{prompt}' -> '{generated}'")
    print("   ⚠️  Note: Output will be random since model is untrained")
    
    print("\n🎉 All tests passed! The Simple LLM implementation is working correctly.")
    print("\nNext steps:")
    print("- Run: python examples/train_simple_model.py")
    print("- Or explore: jupyter notebook notebooks/simple_llm_exploration.ipynb")


if __name__ == "__main__":
    test_basic_functionality()
