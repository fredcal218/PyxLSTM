"""
Main script for training and evaluating depression severity prediction models.

This script is the entry point for the depression severity prediction project,
providing command-line arguments and launching the training process.
"""

import os
from depression_prediction.train import train


def main():
    """Define hyperparameters and run the training process."""
    
    class Config:
        def __init__(self):
            # Data parameters
            self.data_dir = "E-DAIC"
            self.output_dir = "./output"
            # Reducing sequence length to avoid memory issues
            self.max_seq_length = 5000  # Reduced from 30000
            self.stride = 2500         # Reduced from 15000
            
            # Model parameters
            self.hidden_size = 64      # Reduced from 128
            self.num_layers = 1        # Reduced from 2
            self.num_blocks = 1        # Reduced from 2
            self.lstm_type = "slstm"   # Options: "slstm", "mlstm"
            self.dropout = 0.2
            
            # Training parameters
            self.batch_size = 4        # Reduced from 64
            self.learning_rate = 0.001
            self.num_epochs = 50
            self.seed = 42
            self.no_cuda = False
            
            # Memory optimization parameters
            self.gradient_clip = 1.0   # Add gradient clipping
            self.checkpoint_interval = 5  # Save every 5 epochs
            self.mixed_precision = True   # Use mixed precision training
    
    # Create configuration
    config = Config()
    
    # Create output directory if not exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
