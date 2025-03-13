"""
Main script for training and evaluating depression severity prediction models.

This script is the entry point for the depression severity prediction project,
providing command-line arguments and launching the training process.
"""

import os
import torch
from depression_prediction.train import train
from depression_prediction.sequence_utils import process_long_sequence
from depression_prediction.utils import print_memory_stats


def main():
    """Define hyperparameters and run the training process."""
    
    class Config:
        def __init__(self):
            # Data parameters - Optimized for xLSTM's strengths
            self.data_dir = "E-DAIC"
            self.output_dir = "./output"
            self.max_seq_length = 20000  # Increased to leverage xLSTM's long sequence capabilities
            self.stride = 10000          # Larger stride for efficiency
            
            # Model Parameters - Optimized for xLSTM architecture
            self.hidden_size = 96        # Sized for representational capacity
            self.num_layers = 2          # Multiple layers per block for xLSTM benefits
            self.num_blocks = 2          # Multiple blocks to process hierarchical features
            self.lstm_type = "slstm"     # Using scalar LSTM which has better training stability
            self.dropout = 0.3           # Higher dropout for better generalization
            
            # Training parameters - Optimized for stable convergence
            self.batch_size = 4          # Small batch for better gradient updates with long sequences
            self.learning_rate = 0.0003  # Middle ground to avoid local minima but ensure stability
            self.num_epochs = 100        # More epochs for thorough training
            self.seed = 42
            self.no_cuda = False
            
            # Stability optimization
            self.gradient_clip = 0.5     # Conservative clipping to prevent exploding gradients
            self.checkpoint_interval = 5
            self.mixed_precision = True
            
            # Early stopping parameters
            self.early_stopping = True
            self.early_stopping_patience = 12
            self.early_stopping_metric = 'loss'
            
            # Performance optimization
            self.num_workers = min(os.cpu_count(), 2)  # Lower worker count for memory efficiency
            
            # Sequence processing parameters
            self.chunk_long_sequences = True     # Enable processing very long sequences in chunks
            self.max_chunk_length = 5000         # Maximum chunk length for processing
            self.chunk_overlap = 1000            # Overlap between chunks
    
    # Create configuration
    config = Config()
    
    # Print system info
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: No")
    
    # Create output directory if not exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Monitor memory before training
    print_memory_stats()
    
    # Run training
    train(config)
    
    # Monitor memory after training
    print_memory_stats()


if __name__ == "__main__":
    main()
