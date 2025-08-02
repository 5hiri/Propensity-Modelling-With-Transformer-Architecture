"""
Text generation utilities for the language model.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import time


class TextGenerator:
    """Text generation utilities for the language model."""
    
    def __init__(self, model, tokenizer, device: str = "auto"):
        """Initialize the text generator.
        
        Args:
            model: Trained language model
            tokenizer: Tokenizer instance
            device: Device to use for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50,
                     temperature: float = 0.8, top_k: Optional[int] = 50,
                     top_p: Optional[float] = 0.9, do_sample: bool = True,
                     repetition_penalty: float = 1.0) -> str:
        """Generate text given a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated text
        """
        # Encode the prompt
        if hasattr(self.tokenizer, 'encode'):
            # Handle different tokenizer types
            if hasattr(self.tokenizer, 'return_tensors'):  # GPT-2 style tokenizer
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            else:  # Character tokenizer or similar
                tokens = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        else:
            raise ValueError("Tokenizer must have an 'encode' method")
        
        if input_ids.size(1) == 0:
            raise ValueError("Empty input after tokenization")
        
        # Generate tokens
        with torch.no_grad():
            generated_ids = self._generate_tokens(
                input_ids, max_new_tokens, temperature, 
                top_k, top_p, do_sample, repetition_penalty
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def _generate_tokens(self, input_ids: torch.Tensor, max_new_tokens: int,
                        temperature: float, top_k: Optional[int], top_p: Optional[float],
                        do_sample: bool, repetition_penalty: float) -> torch.Tensor:
        """Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated token IDs
        """
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get current sequence length
            current_len = generated.size(1)
            
            # Truncate if exceeding max length
            if current_len >= self.model.max_seq_len:
                input_seq = generated[:, -self.model.max_seq_len:]
            else:
                input_seq = generated
            
            # Forward pass
            outputs = self.model(input_seq, return_dict=True)
            logits = outputs["logits"]
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :].clone()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, repetition_penalty
                )
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p filtering
            if top_p is not None:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample or select next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end of sequence token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                 generated: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits.
        
        Args:
            logits: Next token logits
            generated: Previously generated tokens
            penalty: Repetition penalty factor
            
        Returns:
            Modified logits
        """
        if penalty == 1.0:
            return logits
        
        # Get unique tokens in the generated sequence
        unique_tokens = torch.unique(generated)
        
        # Apply penalty to repeated tokens
        for token in unique_tokens:
            if logits[0, token] > 0:
                logits[0, token] /= penalty
            else:
                logits[0, token] *= penalty
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits.
        
        Args:
            logits: Input logits
            top_k: Number of top tokens to keep
            
        Returns:
            Filtered logits
        """
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits.
        
        Args:
            logits: Input logits
            top_p: Cumulative probability threshold
            
        Returns:
            Filtered logits
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted indices to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def generate_multiple(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            generated = self.generate_text(prompt, **kwargs)
            results.append(generated)
        return results
    
    def interactive_generation(self, max_new_tokens: int = 50,
                             temperature: float = 0.8, top_k: int = 50,
                             top_p: float = 0.9) -> None:
        """Interactive text generation loop.
        
        Args:
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
        """
        print("Interactive Text Generation")
        print("Enter 'quit' to exit")
        print("-" * 40)
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    print("Please enter a non-empty prompt.")
                    continue
                
                # Generate text
                start_time = time.time()
                generated_text = self.generate_text(
                    prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_k=top_k, top_p=top_p
                )
                generation_time = time.time() - start_time
                
                print(f"\nGenerated text:")
                print(f"{generated_text}")
                print(f"\nGeneration time: {generation_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error during generation: {e}")
    
    def compare_generation_strategies(self, prompt: str, max_new_tokens: int = 50) -> Dict[str, str]:
        """Compare different generation strategies.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Dictionary with different generation strategies and their outputs
        """
        strategies = {
            "greedy": {"do_sample": False},
            "sampling_low_temp": {"temperature": 0.5, "do_sample": True},
            "sampling_high_temp": {"temperature": 1.2, "do_sample": True},
            "top_k": {"top_k": 10, "do_sample": True},
            "top_p": {"top_p": 0.7, "do_sample": True},
            "combined": {"temperature": 0.8, "top_k": 50, "top_p": 0.9, "do_sample": True}
        }
        
        results = {}
        for strategy_name, params in strategies.items():
            generated = self.generate_text(prompt, max_new_tokens=max_new_tokens, **params)
            results[strategy_name] = generated
        
        return results
