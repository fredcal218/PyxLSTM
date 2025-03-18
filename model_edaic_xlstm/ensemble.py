import torch
import torch.nn as nn
from model import EDAIC_LSTM
import time

class DepressionEnsemble(nn.Module):
    """
    Memory-optimized ensemble model for depression severity prediction that combines:
    1. A binary classifier for depression/no-depression detection (threshold at PHQ-8 = 10)
    2. A regressor specialized for low range scores (PHQ-8: 0-9)
    3. A regressor specialized for high range scores (PHQ-8: 10-24)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, seq_length=600):
        super(DepressionEnsemble, self).__init__()
        
        # Binary classifier: predicts if depression score is >= 10
        self.binary_classifier = EDAIC_LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=1,
            seq_length=seq_length
        )
        
        # Low-range regressor: specialized for PHQ-8 scores 0-9
        self.low_regressor = EDAIC_LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=1,
            seq_length=seq_length
        )
        
        # High-range regressor: specialized for PHQ-8 scores 10-24
        self.high_regressor = EDAIC_LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=1,
            seq_length=seq_length
        )
        
        # Sigmoid activation for binary classifier output
        self.sigmoid = nn.Sigmoid()
        
        # Confidence threshold for using high-range model
        self.confidence_threshold = 0.5
        
    def forward(self, x):
        """
        Forward pass through the ensemble.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, seq_len, features]
            
        Returns:
            tuple: (depression_score, depression_probability, model_used)
                - depression_score: Predicted PHQ-8 score
                - depression_probability: Probability of clinical depression (PHQ-8 >= 10)
                - model_used: Indicator of which regressor was used (0=low, 1=high)
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # First run binary classifier to determine depression probability
        with torch.amp.autocast(device_type='cuda', enabled=True):
            depression_logit = self.binary_classifier(x)
            
            # Check for NaN or Inf values in logits
            if torch.isnan(depression_logit).any() or torch.isinf(depression_logit).any():
                depression_logit = torch.nan_to_num(depression_logit, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply sigmoid and clamp to valid range [0,1]
            depression_prob = self.sigmoid(depression_logit)
            depression_prob = torch.clamp(depression_prob, min=0.0, max=1.0)
            
            # Clear memory
            torch.cuda.empty_cache()
            
            # Create a consistent output shape for all tensors
            combined_scores = torch.zeros_like(depression_prob)
            
            # Convert masks to proper shapes for indexing
            # Reshape to [batch_size] for proper batch-dimension indexing
            high_mask_flat = (depression_prob >= self.confidence_threshold).view(-1)
            low_mask_flat = ~high_mask_flat
            
            # Keep original 2D masks for assigning results
            high_mask = (depression_prob >= self.confidence_threshold)
            low_mask = ~high_mask
            
            # Process samples by group (low/high)
            # Only process if we have samples in that category
            if low_mask_flat.any():
                # Extract samples with low scores using the flat mask
                low_inputs = x[low_mask_flat]
                
                # Handle single sample edge case
                if len(low_inputs.shape) == 2:
                    low_inputs = low_inputs.unsqueeze(0)
                
                # Get predictions
                low_score = self.low_regressor(low_inputs)
                
                # Assign to combined scores - reshape low_score to match combined_scores[low_mask]
                combined_scores[low_mask] = low_score.view(-1)
                
                # Clean up
                del low_inputs, low_score
                torch.cuda.empty_cache()
            
            if high_mask_flat.any():
                # Extract samples with high scores using the flat mask
                high_inputs = x[high_mask_flat]
                
                # Handle single sample edge case
                if len(high_inputs.shape) == 2:
                    high_inputs = high_inputs.unsqueeze(0)
                
                # Get predictions
                high_score = self.high_regressor(high_inputs)
                
                # Fix: Reshape high_score to match the shape of combined_scores[high_mask]
                # The diagnostic info shows high_score is [8, 1] but we need [8]
                combined_scores[high_mask] = high_score.view(-1)
                
                # Clean up
                del high_inputs, high_score
                torch.cuda.empty_cache()
            
            # Create mask for binary classification (1=high range, 0=low range)
            high_range_mask = (depression_prob >= self.confidence_threshold).float()
            
            # Safety check for NaN/Inf values in outputs
            if torch.isnan(combined_scores).any() or torch.isinf(combined_scores).any():
                combined_scores = torch.nan_to_num(combined_scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            if torch.isnan(depression_prob).any() or torch.isinf(depression_prob).any():
                depression_prob = torch.nan_to_num(depression_prob, nan=0.5, posinf=1.0, neginf=0.0)
                depression_prob = torch.clamp(depression_prob, min=0.0, max=1.0)
        
        # Return all predictions
        return combined_scores, depression_prob, high_range_mask
    
    def calculate_accuracy(self, predictions, targets, threshold=10.0):
        """Pass-through to the binary accuracy calculation"""
        # We only need the first element (combined scores) for this
        return EDAIC_LSTM.calculate_accuracy(self, predictions[0], targets, threshold)
    
    def calculate_overall_accuracy(self, predictions, targets, tolerance=2.0):
        """Pass-through to the overall accuracy calculation"""
        return EDAIC_LSTM.calculate_overall_accuracy(self, predictions[0], targets, tolerance)
    
    def calculate_per_level_accuracy(self, predictions, targets):
        """Pass-through to the per-level accuracy calculation"""
        return EDAIC_LSTM.calculate_per_level_accuracy(self, predictions[0], targets)
    
    def calculate_score_tolerance_accuracy(self, predictions, targets, tolerance=2.0):
        """Pass-through to the score tolerance accuracy calculation"""
        return EDAIC_LSTM.calculate_score_tolerance_accuracy(self, predictions[0], targets, tolerance)
