"""
Main script for training and evaluating depression severity prediction models.

This script is the entry point for the depression severity prediction project,
providing command-line arguments and launching the training process.
"""

import os
import torch
from depression_prediction.train import train


def main():
    """Define hyperparameters and run the training process."""
    
    class Config:
        def __init__(self):
            # Data parameters
            self.data_dir = "E-DAIC"
            self.output_dir = "./output"
            self.max_seq_length = 30000
            self.stride = 25000
            
            # Model Parameters
            self.hidden_size = 64
            self.num_layers = 2
            self.num_blocks = 2
            self.lstm_type = "slstm"
            self.dropout = 0.2
            
            # Training parameters
            self.batch_size = 32
            self.learning_rate = 0.00005
            self.num_epochs = 50
            self.seed = 42
            self.no_cuda = False
            
            # Memory and stability optimization
            self.gradient_clip = 0.5
            self.checkpoint_interval = 1
            self.mixed_precision = True
            
            # Early stopping parameters
            self.early_stopping = True
            self.early_stopping_patience = 5  
            self.early_stopping_metric = 'mae'  # Can be 'loss', 'mae', 'rmse', 'r2'
            
            # Performance optimization
            self.num_workers = min(os.cpu_count(), 4)
    
    # Create configuration
    config = Config()
    
    # Print system info
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    else:
        print("CUDA available: No")
    
    # Create output directory if not exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
