import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class DepBinaryClassifier(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=2, dropout=0.5, seq_length=150,
                 feature_names=None):
        """
        Binary classification model for E-DAIC depression detection with interpretability
        
        Args:
            input_size (int): Size of input features (default: 75 - combined pose, gaze and AUs)
            hidden_size (int): LSTM hidden state size
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            seq_length (int): Length of input sequences
            feature_names (list): Names of input features for interpretability
        """
        super(DepBinaryClassifier, self).__init__()

        # Store feature names for interpretability
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(input_size)]
        self.input_size = input_size
        
        # Project raw input features into the hidden dimension
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Apply dropout after input projection
        self.input_dropout = nn.Dropout(dropout)

        # Configure the xLSTM stack for non-language applications
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                    dropout=dropout,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                    dropout=dropout,
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=seq_length,
            num_blocks=num_layers,
            embedding_dim=hidden_size,
            slstm_at=[1],
            dropout=dropout,
            bias=True,
        )
        
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.xlstm_stack.reset_parameters()
        
        # Output layer for binary classification
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        # For storing feature importance information - initialize as empty dict
        self.feature_importances = {}
        self.last_attention_weights = None

    def forward(self, x, return_intermediates=False):
        """
        Forward pass with option to return intermediate activations
        
        Args:
            x (torch.Tensor): Input features [batch, seq_length, input_size]
            return_intermediates (bool): Whether to return intermediate activations for interpretability
            
        Returns:
            torch.Tensor or tuple: Binary classification logits, or tuple with intermediates
        """
        # x shape: [batch, seq_length, input_size]
        batch_size, seq_len, _ = x.shape
        
        # Ensure input dimensions match expected dimensions
        if seq_len > self.xlstm_stack.config.context_length:
            x = x[:, :self.xlstm_stack.config.context_length, :]
        elif seq_len < self.xlstm_stack.config.context_length:
            padding = torch.zeros(batch_size, 
                                 self.xlstm_stack.config.context_length - seq_len, 
                                 x.shape[-1], 
                                 device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Project and apply dropout
        x_proj = self.input_dropout(self.input_proj(x))
        
        # Process through xLSTM stack
        out_seq = self.xlstm_stack(x_proj)  # Output shape: [batch, seq_length, hidden_size]
        
        # Use the last timestep for classification
        out_last = out_seq[:, -1, :]
        out_last = self.dropout(out_last)
        logits = self.fc(out_last)
        
        # Return raw logits (no sigmoid) for use with BCEWithLogitsLoss
        if return_intermediates:
            return logits, {'input': x, 'projected': x_proj, 'sequence_out': out_seq, 'final_out': out_last}
        return logits
    
    def predict_proba(self, x):
        """
        Get probability predictions by applying sigmoid to logits
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Probability predictions
        """
        logits = self(x)
        return torch.sigmoid(logits)
    
    def calculate_metrics(self, logits, targets, threshold=0.0):
        """
        Calculate binary classification metrics
        
        Args:
            logits (torch.Tensor): Model's output logits
            targets (torch.Tensor): Ground truth binary labels
            threshold (float): Classification threshold on logits (0.0 = default decision boundary)
            
        Returns:
            dict: Dictionary containing accuracy, precision, recall, and F1 score
        """
        # Convert logits to binary predictions using threshold
        probs = torch.sigmoid(logits)
        binary_preds = (logits > threshold).float()  # or (probs > 0.5)
        
        # Calculate metrics
        tp = ((binary_preds == 1) & (targets == 1)).float().sum()
        fp = ((binary_preds == 1) & (targets == 0)).float().sum()
        tn = ((binary_preds == 0) & (targets == 0)).float().sum()
        fn = ((binary_preds == 0) & (targets == 1)).float().sum()
        
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }
    
    def get_attention_weights(self):
        """
        Extract attention weights from the xLSTM model
        
        Returns:
            list: Attention weights from each layer
        """
        attention_weights = []
        
        # Extract attention weights from mLSTM and sLSTM layers
        for block_idx, block in enumerate(self.xlstm_stack.blocks):
            # Check if the block has mlstm
            if hasattr(block, 'mlstm') and hasattr(block.mlstm, 'attn'):
                if hasattr(block.mlstm.attn, 'last_attn_weights'):
                    attention_weights.append({
                        'layer': f'block_{block_idx}_mlstm',
                        'weights': block.mlstm.attn.last_attn_weights.detach()
                    })
            
            # Check if the block has slstm
            if hasattr(block, 'slstm') and hasattr(block.slstm, 'attn'):
                if hasattr(block.slstm.attn, 'last_attn_weights'):
                    attention_weights.append({
                        'layer': f'block_{block_idx}_slstm',
                        'weights': block.slstm.attn.last_attn_weights.detach()
                    })
        
        self.last_attention_weights = attention_weights
        return attention_weights
    
    def gradient_feature_importance(self, x, target_class=1):
        """
        Calculate feature importance using gradient-based attribution
        
        Args:
            x (torch.Tensor): Input features [batch, seq_len, feature_dim]
            target_class (int): Target class (1=depressed, 0=not depressed)
            
        Returns:
            dict: Importance scores for each input feature
        """
        # Set model to eval mode
        self.eval()
        x_input = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self(x_input)
        
        # Select target based on class - use logits directly
        if target_class == 1:
            # For positive class: maximize logits (equivalent to minimizing -logits)
            target = logits  # We want to maximize this
        else:
            # For negative class: minimize logits (equivalent to maximizing -logits)
            target = -logits  # We want to maximize this (same as minimizing logits)
        
        # Compute gradients w.r.t inputs
        gradients = torch.autograd.grad(
            outputs=target.sum(),
            inputs=x_input,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Feature importance is the mean absolute gradient per feature
        # Average over batch and sequence dimensions
        feature_importance = torch.mean(torch.abs(gradients), dim=(0, 1)).detach().cpu().numpy()
        
        # Normalize importance scores
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        # Store feature importance
        if 'global' not in self.feature_importances:
            self.feature_importances['global'] = {}
            
        self.feature_importances['global']['gradient_based'] = feature_importance
        
        # Map feature importance to feature names
        importance_dict = {
            self.feature_names[i]: feature_importance[i] 
            for i in range(min(len(self.feature_names), len(feature_importance)))
        }
        
        return importance_dict
    
    def integrated_gradients(self, x, baseline=None, steps=50, target_class=1):
        """
        Calculate feature importance using integrated gradients
        
        Args:
            x (torch.Tensor): Input features [batch, seq_len, feature_dim]
            baseline (torch.Tensor): Baseline input (zeros by default)
            steps (int): Number of steps for path integral
            target_class (int): Target class (1=depressed, 0=not depressed)
            
        Returns:
            dict: Importance scores for each feature
        """
        # Set model to eval mode and ensure we're working with PyTorch tensors
        self.eval()
        x = x.detach()
        
        # Create baseline if not provided (zeros)
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        # Debug print
        print(f"Input shape: {x.shape}, Range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        
        # Scale inputs along path from baseline to input
        scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)]
        
        # Calculate gradients at each step
        total_gradients = torch.zeros_like(x)
        
        for i, scaled_input in enumerate(scaled_inputs):
            scaled_input = scaled_input.clone().detach().requires_grad_(True)
            
            # Forward pass
            logits = self(scaled_input)
            
            # Determine target based on class
            if target_class == 1:
                target = logits  # We want positive logits for "depressed" class
            else:
                target = -logits  # We want negative logits for "not depressed" class
            
            # Compute gradients for this step
            grad = torch.autograd.grad(
                outputs=target.sum(),
                inputs=scaled_input,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Accumulate gradients
            total_gradients += grad.detach()
            
            # Debug every 10 steps
            if i % 10 == 0:
                print(f"Step {i}/{steps}: Grad range [{grad.min().item():.4f}, {grad.max().item():.4f}]")
        
        # Average the gradients
        avg_gradients = total_gradients / (steps + 1)
        
        # Element-wise multiply average gradients with the input-baseline difference
        attributions = (x - baseline) * avg_gradients
        
        # Sum over batch and sequence dimensions
        feature_importance = torch.mean(torch.abs(attributions), dim=(0, 1)).cpu().numpy()
        
        # Normalize if possible
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        # Store in feature importances
        if 'global' not in self.feature_importances:
            self.feature_importances['global'] = {}
        
        self.feature_importances['global']['integrated_gradients'] = feature_importance
        
        # Debug print overall importance results
        print(f"Feature importance range: [{feature_importance.min():.6f}, {feature_importance.max():.6f}]")
        print(f"Sum of importance: {np.sum(feature_importance):.6f}")
        
        # Map to feature names
        importance_dict = {
            self.feature_names[i]: feature_importance[i] 
            for i in range(min(len(self.feature_names), len(feature_importance)))
        }
        
        # Debug print top features
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 features:", top_features)
        
        return importance_dict
    
    def instance_feature_importance(self, x):
        """
        Calculate instance-specific feature importance
        
        Args:
            x (torch.Tensor): Single instance input [1, seq_len, feature_dim]
            
        Returns:
            dict: Feature importance for this specific instance
        """
        # Get prediction
        with torch.no_grad():
            logits = self(x)
            prob = torch.sigmoid(logits)
            pred_class = 1 if prob.item() > 0.5 else 0
        
        # Calculate feature importance for predicted class
        print(f"Instance prediction: class={pred_class}, probability={prob.item():.4f}")
        importance = self.integrated_gradients(x, target_class=pred_class)
        
        # Store instance importance
        if 'instance' not in self.feature_importances:
            self.feature_importances['instance'] = {}
        
        instance_id = len(self.feature_importances['instance'])
        self.feature_importances['instance'][instance_id] = {
            'prediction': prob.item(),
            'predicted_class': pred_class,
            'importance': importance
        }
        
        return {
            'prediction': prob.item(),
            'predicted_class': pred_class,
            'importance': importance
        }
    
    def visualize_feature_importance(self, importance_dict=None, top_k=20, save_path=None, 
                                   title="Top Feature Importance for Depression Detection"):
        """
        Visualize feature importance
        
        Args:
            importance_dict (dict): Dictionary mapping feature names to importance scores
            top_k (int): Number of top features to display
            save_path (str): Path to save visualization
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Feature importance visualization
        """
        if importance_dict is None:
            if self.feature_importances is None or 'global' not in self.feature_importances:
                raise ValueError("No feature importance data available")
            
            # Use integrated gradients by default if available
            if 'integrated_gradients' in self.feature_importances['global']:
                importance_array = self.feature_importances['global']['integrated_gradients']
            elif 'gradient_based' in self.feature_importances['global']:
                importance_array = self.feature_importances['global']['gradient_based']
            else:
                importance_array = next(iter(self.feature_importances['global'].values()))
            
            # Debug print importance array stats
            print(f"Importance array: min={importance_array.min():.6f}, max={importance_array.max():.6f}, mean={importance_array.mean():.6f}")
            
            importance_dict = {
                self.feature_names[i]: importance_array[i]
                for i in range(min(len(self.feature_names), len(importance_array)))
            }
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k features
        top_features = sorted_features[:top_k]
        feature_names = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(feature_names)), importance_values, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Importance')
        plt.title(title)
        
        # Highlight AUs of interest
        highlighted_aus = ['AU01', 'AU04', 'AU05', 'AU06', 'AU07', 'AU12', 'AU15']
        for i, feature in enumerate(feature_names):
            for au in highlighted_aus:
                if au in feature:
                    bars[i].set_color('red')
                    break
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def visualize_attention(self, layer_idx=0, head_idx=0, save_path=None):
        """
        Visualize attention weights from a specific layer and head
        
        Args:
            layer_idx (int): Index of the layer to visualize
            head_idx (int): Index of the attention head to visualize
            save_path (str): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Attention visualization
        """
        if self.last_attention_weights is None:
            raise ValueError("No attention weights available. Run get_attention_weights() first.")
        
        if layer_idx >= len(self.last_attention_weights):
            raise ValueError(f"Layer index {layer_idx} out of range. Only {len(self.last_attention_weights)} layers available.")
        
        # Get attention weights for the specified layer
        layer_info = self.last_attention_weights[layer_idx]
        layer_name = layer_info['layer']
        attn_weights = layer_info['weights']
        
        # For multi-head attention, select the head
        if len(attn_weights.shape) == 4:  # [batch, heads, seq, seq]
            if head_idx >= attn_weights.shape[1]:
                raise ValueError(f"Head index {head_idx} out of range. Only {attn_weights.shape[1]} heads available.")
            # Select head and average over batch
            attn_weights = attn_weights[:, head_idx].mean(0)
        else:
            # Average over batch
            attn_weights = attn_weights.mean(0)
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights.cpu().numpy(), cmap='viridis')
        plt.title(f'Attention Weights: {layer_name}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_clinical_au_importance(self):
        """
        Extract importance of clinically significant Action Units
        
        Returns:
            dict: Importance scores for clinically significant AUs
        """
        # Key AUs from literature
        clinical_aus = {
            'AU01': 'Inner Brow Raiser',
            'AU04': 'Brow Lowerer',
            'AU05': 'Upper Lid Raiser',
            'AU06': 'Cheek Raiser',
            'AU07': 'Lid Tightener',
            'AU12': 'Lip Corner Puller',
            'AU15': 'Lip Corner Depressor'
        }
        
        if self.feature_importances is None or 'global' not in self.feature_importances:
            raise ValueError("No feature importance data available. Run gradient_feature_importance() first.")
        
        # Get most recent feature importance
        if 'integrated_gradients' in self.feature_importances['global']:
            importance_array = self.feature_importances['global']['integrated_gradients']
        else:
            importance_array = next(iter(self.feature_importances['global'].values()))
        
        # Extract importance for clinical AUs
        clinical_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            for au_code, au_name in clinical_aus.items():
                if au_code in feature_name:
                    if i < len(importance_array):
                        clinical_importance[f"{au_code} ({au_name})"] = importance_array[i]
        
        # Sort by importance
        clinical_importance = {k: v for k, v in 
                             sorted(clinical_importance.items(), 
                                   key=lambda item: item[1], 
                                   reverse=True)}
        
        return clinical_importance
