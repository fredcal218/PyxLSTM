import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """
    Attention mechanism to weigh the importance of input features dynamically.
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: [batch, seq_len, hidden_size]
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)  # [batch, seq_len, 1]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch, hidden_size]
        return context_vector, attention_weights

class BiLSTM_Attention(nn.Module):
    def __init__(self, input_size=31, hidden_size=64, num_layers=3, dropout=0.3, num_classes=1, seq_length=150):
        """
        Bidirectional LSTM model for E-DAIC depression detection using pose, gaze, and action unit data
        with an attention mechanism.
        
        Args:
            input_size (int): Size of input features (default: 31 - combined pose, gaze and AUs)
            hidden_size (int): LSTM hidden state size (default: 64)
            num_layers (int): Number of LSTM layers (default: 3)
            dropout (float): Dropout probability (default: 0.3)
            num_classes (int): Number of output classes (1 for depression score regression)
            seq_length (int): Length of input sequences (used for padding/truncating)
        """
        super(BiLSTM_Attention, self).__init__()

        # Project raw input features into the hidden dimension
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Apply dropout after input projection
        self.input_dropout = nn.Dropout(dropout)

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)  # * 2 for bidirectional
        
        # Output layer: Map attention output to regression target
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # Store sequence length for padding/truncating
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: [batch, seq_length, input_size]
        batch_size, seq_len, _ = x.shape
        
        # Ensure input dimensions match expected dimensions
        if seq_len > self.seq_length:
            # Truncate if input sequence is too long
            x = x[:, :self.seq_length, :]
        elif seq_len < self.seq_length:
            # Pad with zeros if input sequence is too short
            padding = torch.zeros(batch_size, 
                                 self.seq_length - seq_len, 
                                 x.shape[-1], 
                                 device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Project and apply dropout
        x_proj = self.input_dropout(self.input_proj(x))  # Shape: [batch, seq_length, hidden_size]
        
        # Process through bidirectional LSTM
        lstm_output, _ = self.lstm(x_proj)  # Output shape: [batch, seq_length, hidden_size*2]
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(lstm_output)
        
        # Apply dropout to context vector
        context_vector = self.dropout(context_vector)
        
        # Final prediction
        output = self.fc(context_vector)
        
        return output

    # Basic metric calculation functions for evaluation
    def calculate_rmse(self, predictions, targets):
        """Calculate RMSE between predictions and targets"""
        return torch.sqrt(F.mse_loss(predictions, targets))
    
    def calculate_mae(self, predictions, targets):
        """Calculate MAE between predictions and targets"""
        return F.l1_loss(predictions, targets)
        
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
    
    def calculate_overall_accuracy(self, predictions, targets, tolerance=2.0):
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
