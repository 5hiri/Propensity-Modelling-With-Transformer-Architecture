"""
Training utilities and trainer class.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
from typing import Dict, Optional, Any, List
from pathlib import Path
import matplotlib.pyplot as plt


class LMTrainer:
    """Trainer class for language model training."""
    
    def __init__(self, model: nn.Module, config: Any, device: str = "auto"):
        """Initialize the trainer.
        
        Args:
            model: The language model to train
            config: Configuration object with training parameters
            device: Device to use for training ("auto", "cpu", "cuda")
        """
        self.model = model
        self.config = config
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Create directories
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the language modeling loss.
        
        Args:
            batch: Batch of data containing input_ids, attention_mask, and labels
            
        Returns:
            Loss tensor
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask, return_dict=True)
        logits = outputs["logits"]
        
        # Compute cross-entropy loss
        # Reshape for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update global step
            self.global_step += 1
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(os.path.dirname(filepath), "best_model.pt")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'], label='Training Loss')
        ax1.plot(epochs, self.training_history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(epochs, self.training_history['learning_rate'], label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        print(f"Starting training for {self.config.max_epochs} epochs...")
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate if validation loader is provided
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Check if this is the best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
                is_best = train_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = train_loss
            
            # Update learning rate
            self.scheduler.step()
            
            # Record training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss if val_loss is not None else train_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or epoch == self.config.max_epochs - 1:
                checkpoint_path = os.path.join(
                    self.config.model_save_path,
                    f"checkpoint_epoch_{epoch}.pt"
                )
                self.save_checkpoint(checkpoint_path, is_best)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            print("-" * 50)
        
        print("Training completed!")
        
        # Plot training history
        plot_path = os.path.join(self.config.log_dir, "training_history.png")
        self.plot_training_history(plot_path)
