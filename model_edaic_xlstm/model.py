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
    
    def calculate_overall_accuracy(self, predictions, targets, tolerance=1.0):
        """
        Calculate overall accuracy for depression scores with tolerance.
        
        Args:
            predictions (torch.Tensor): Model's predicted depression scores
            targets (torch.Tensor): Ground truth depression scores
            tolerance (float): Tolerance for considering a prediction correct
                              (prediction is correct if |pred - target| <= tolerance)
            
        Returns:
            float: Overall accuracy score
        """
        # Consider prediction correct if within tolerance range of target
        correct_predictions = (torch.abs(predictions - targets) <= tolerance).float().sum()
        total = targets.numel()
        
        return correct_predictions / total
    
    def calculate_per_level_accuracy(self, predictions, targets):
        """
        Calculate accuracy for each PHQ-8 depression severity level.
        
        PHQ-8 Score ranges:
        0-4: None or minimal (level 0)
        5-9: Mild (level 1)
        10-14: Moderate (level 2)
        15-19: Moderately severe (level 3)
        20-24: Severe (level 4)
        
        Args:
            predictions (torch.Tensor): Model's predicted depression scores
            targets (torch.Tensor): Ground truth depression scores
            
        Returns:
            tuple: (overall_accuracy, level_accuracies)
                - overall_accuracy: float, accuracy across all categories
                - level_accuracies: list of floats, accuracy for each severity level
        """
        # Define PHQ level thresholds and names
        level_thresholds = torch.tensor([0, 5, 10, 15, 20, 25], device=predictions.device)
        level_names = ['None/Minimal', 'Mild', 'Moderate', 'Mod. Severe', 'Severe']
        
        # Convert predictions to levels
        pred_levels = torch.zeros_like(predictions, dtype=torch.long)
        target_levels = torch.zeros_like(targets, dtype=torch.long)
        
        # Assign levels based on thresholds
        for i in range(len(level_thresholds)-1):
            pred_levels[(predictions >= level_thresholds[i]) & (predictions < level_thresholds[i+1])] = i
            target_levels[(targets >= level_thresholds[i]) & (targets < level_thresholds[i+1])] = i
        
        # Handle edge case for maximum score
        pred_levels[predictions >= level_thresholds[-1]] = len(level_thresholds) - 2
        target_levels[targets >= level_thresholds[-1]] = len(level_thresholds) - 2
        
        # Calculate overall accuracy
        correct = (pred_levels == target_levels).float().sum()
        total = target_levels.numel()
        overall_accuracy = (correct / total).item() if total > 0 else 0.0
        
        # Calculate accuracy for each level
        level_accuracies = []
        for i in range(len(level_names)):
            # Get samples where the true level is i
            level_mask = (target_levels == i)
            level_count = level_mask.sum().item()
            
            if level_count > 0:
                # Calculate accuracy for this level
                level_correct = ((pred_levels == i) & level_mask).float().sum()
                level_accuracy = (level_correct / level_count).item()
            else:
                level_accuracy = 0.0  # No samples for this level
                
            level_accuracies.append(level_accuracy)
        
        return overall_accuracy, level_accuracies