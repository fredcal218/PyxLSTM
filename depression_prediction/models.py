"""
Models for depression severity prediction using pose, gaze, and AU features.

This module provides models based on the xLSTM architecture for predicting
depression severity using the PHQ-8 scale.
"""

import torch
import torch.nn as nn
from xLSTM import xLSTMBlock


class DepressionPredictor(nn.Module):
    """
    Depression severity predictor using xLSTM architecture.
    
    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state in LSTM.
        num_layers (int): Number of LSTM layers in each block.
        num_blocks (int): Number of xLSTM blocks.
        dropout (float, optional): Dropout probability. Default: 0.2.
        lstm_type (str, optional): Type of LSTM to use ('slstm' or 'mlstm'). Default: 'slstm'.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_blocks, 
                 dropout=0.2, lstm_type="slstm"):
        super(DepressionPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        
        print(f"  Creating feature projection layer: {input_size} -> {hidden_size}")
        # Feature projection layer with gradient stabilization
        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Add layer normalization for stability
            nn.ReLU()
        )
        
        # xLSTM blocks
        print(f"  Creating {num_blocks} {lstm_type} blocks with {num_layers} layers each")
        self.blocks = nn.ModuleList()
        self.lstm_block_count = 0  # Track actual number of LSTM blocks
        
        for i in range(num_blocks):
            print(f"    Creating block {i+1}/{num_blocks}")
            self.blocks.append(
                xLSTMBlock(hidden_size, hidden_size, num_layers, dropout, lstm_type)
            )
            self.lstm_block_count += 1
            
            # Add stabilizing layers after each block
            if i < num_blocks - 1:  # Don't add after the last block
                self.blocks.append(nn.LayerNorm(hidden_size))
        
        print("  Creating attention layer for sequence aggregation")
        # Attention layer for sequence aggregation with stability improvements
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),  # More stable than ReLU for attention
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        print("  Creating PHQ-8 regression head")
        # Regression head for PHQ-8 prediction with gradient stabilization
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        print("  Model architecture completed")

    def forward(self, x, hidden_states=None):
        """
        Forward pass of the depression predictor.
        
        Args:
            x (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_states (list of tuples, optional): Initial hidden states for each block.
        
        Returns:
            tuple: Predicted PHQ-8 score and final hidden states.
        """
        batch_size, seq_length, _ = x.size()
        
        # Project input features with stabilized gradient flow
        x = self.feature_proj(x)
        
        # Initialize hidden states if not provided - fixing the error
        if hidden_states is None:
            hidden_states = [None] * self.lstm_block_count
        
        # Process through xLSTM blocks
        block_idx = 0
        for i in range(len(self.blocks)):
            module = self.blocks[i]
            if isinstance(module, xLSTMBlock):
                x, hidden_states[block_idx] = module(x, hidden_states[block_idx])
                block_idx += 1
            else:
                # It's a normalization layer
                x = module(x)
        
        # Apply attention to aggregate sequence information
        attention_weights = self.attention(x)
        
        # Apply attention with numerical stability check
        weighted_x = x * attention_weights
        
        # Check for NaN/Inf values
        if torch.isnan(weighted_x).any() or torch.isinf(weighted_x).any():
            # If found, replace with zeros
            print("Warning: NaN/Inf detected in attention output, replacing with zeros")
            weighted_x = torch.nan_to_num(weighted_x, nan=0.0, posinf=0.0, neginf=0.0)
        
        context_vector = torch.sum(weighted_x, dim=1)
        
        # Predict PHQ-8 score with numerical stability
        phq8_score = self.regressor(context_vector)
        
        # Final safety check for NaN/Inf values
        if torch.isnan(phq8_score).any() or torch.isinf(phq8_score).any():
            # If found, replace with zeros
            print("Warning: NaN/Inf detected in model output, replacing with zeros")
            phq8_score = torch.zeros_like(phq8_score)
        
        return phq8_score, hidden_states


def create_model(config):
    """
    Create a depression predictor model based on configuration.
    
    Args:
        config (dict): Model configuration.
    
    Returns:
        DepressionPredictor: The configured model.
    """
    print(f"Creating DepressionPredictor with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return DepressionPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_blocks=config['num_blocks'],
        dropout=config.get('dropout', 0.2),
        lstm_type=config.get('lstm_type', 'slstm')
    )
