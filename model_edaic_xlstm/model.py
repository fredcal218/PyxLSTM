import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class EDAIC_LSTM(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=2, dropout=0.5, num_classes=1, seq_length=150):
        """
        LSTM model for E-DAIC depression detection using pose, gaze, and action unit data
        
        Args:
            input_size (int): Size of input features (default: 75 - combined pose, gaze and AUs)
            hidden_size (int): LSTM hidden state size
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            num_classes (int): Number of output classes (1 for depression score regression)
            seq_length (int): Length of input sequences
        """
        super(EDAIC_LSTM, self).__init__()

        # Project raw input features into the hidden dimension
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Apply dropout after input projection
        self.input_dropout = nn.Dropout(dropout)

        # Choose backend based on device availability
        device_backend = "cuda" if torch.cuda.is_available() else "vanilla"

        # Configure the xLSTM stack for non-language applications
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                    dropout=dropout,  # Add dropout to mLSTM layers
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                    dropout=dropout,  # Add dropout to sLSTM layers
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=seq_length,
            num_blocks=num_layers,
            embedding_dim=hidden_size,
            slstm_at=[1],  # Use sLSTM for the second layer
            dropout=dropout,  # Overall dropout for the stack
            bias=True,  # Enable bias parameters
        )
        
        self.xlstm_stack = xLSTMBlockStack(cfg)
        
        # Initialize parameters explicitly
        self.xlstm_stack.reset_parameters()

        # Output layer: Map last timestep output to regression targets
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch, seq_length, input_size]
        batch_size, seq_len, _ = x.shape
        
        # Ensure input dimensions match expected dimensions
        if seq_len > self.xlstm_stack.config.context_length:
            # Truncate if input sequence is too long
            x = x[:, :self.xlstm_stack.config.context_length, :]
        elif seq_len < self.xlstm_stack.config.context_length:
            # Pad with zeros if input sequence is too short
            padding = torch.zeros(batch_size, 
                                 self.xlstm_stack.config.context_length - seq_len, 
                                 x.shape[-1], 
                                 device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Project and apply dropout
        x_proj = self.input_dropout(self.input_proj(x))  # Shape: [batch, seq_length, hidden_size]
        
        # Process through xLSTM stack
        out = self.xlstm_stack(x_proj)  # Output shape: [batch, seq_length, hidden_size]
        
        # Take last timestep for prediction
        out_last = out[:, -1, :]
        out_last = self.dropout(out_last)
        output = self.fc(out_last)
        
        return output

    def calculate_accuracy(self, predictions, targets, threshold=10.0):
        """
        Calculate binary accuracy for depression scores based on clinical threshold.
        
        Args:
            predictions (torch.Tensor): Model's predicted depression scores
            targets (torch.Tensor): Ground truth depression scores
            threshold (float): Clinical threshold for depression (10.0)
                               Scores > 10 indicate depression, scores <= 10 indicate no depression
            
        Returns:
            float: Accuracy score
        """
        # Convert regression outputs to binary predictions using clinical threshold
        binary_preds = (predictions > threshold).float()
        binary_targets = (targets > threshold).float()
        
        # Calculate accuracy
        correct = (binary_preds == binary_targets).float().sum()
        total = binary_targets.numel()
        
        return correct / total