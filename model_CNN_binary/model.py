import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleCNNEncoder(nn.Module):
    """Simplified CNN encoder for all features combined"""
    def __init__(self, input_size, hidden_size, seq_length, dropout=0.5, num_conv_layers=3):
        super(SimpleCNNEncoder, self).__init__()
        
        # Initial projection to match hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # Create a stack of 1D convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            # Calculate proper padding to maintain sequence length
            dilation = 2**i
            kernel_size = 3
            padding = (kernel_size - 1) * dilation // 2
            
            # Each layer: conv -> batchnorm -> relu -> dropout
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation
                ),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_layer)
        
        # Global average pooling (simplified from attention pooling)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """Forward pass
        Args:
            x: Input tensor [batch_size, seq_length, input_size]
        Returns:
            Processed features [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden size
        x_proj = self.input_dropout(self.input_proj(x))  # [batch, seq_len, hidden_size]
        
        # Transpose for conv1d: [batch, channels, seq_len]
        x_conv = x_proj.transpose(1, 2)  # [batch, hidden_size, seq_len]
        
        # Apply convolutional layers with residual connections
        for conv_layer in self.conv_layers:
            x_conv = conv_layer(x_conv) + x_conv  # Residual connection
            
        # Transpose back to [batch, seq_len, hidden_size]
        x_out = x_conv.transpose(1, 2)
        
        return x_out
    
    def pool_sequence(self, x):
        """Pool sequence into a single vector using average pooling
        Args:
            x: Tensor [batch_size, seq_length, hidden_size]
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Transpose to [batch, hidden_size, seq_len] for pooling
        x_t = x.transpose(1, 2)
        
        # Apply global average pooling
        pooled = self.global_pool(x_t).squeeze(-1)  # [batch, hidden_size]
        
        return pooled

class FeatureGroupBalancer(nn.Module):
    """
    Module to balance the influence of different feature groups (pose, AU, gaze)
    by applying group-specific scaling and learnable weights.
    """
    def __init__(self, feature_names, pose_scaling_factor=0.5):
        super(FeatureGroupBalancer, self).__init__()
        
        # Identify feature group indices
        self.pose_indices, self.gaze_indices, self.au_indices = self._identify_feature_groups(feature_names)
        
        # Set feature group scaling factors (fixed)
        self.pose_scaling_factor = pose_scaling_factor
        
        # Create learnable weights for each feature group (initialized to balance groups)
        self.feature_group_weights = nn.Parameter(torch.ones(3))  # [pose, gaze, AU]
        
        # Store feature names for reference
        self.feature_names = feature_names
        
    def _identify_feature_groups(self, feature_names):
        """Identify indices for each feature group"""
        pose_indices = []
        gaze_indices = []
        au_indices = []
        
        for i, name in enumerate(feature_names):
            if name.startswith('pose_'):
                pose_indices.append(i)
            elif name.startswith('gaze_'):
                gaze_indices.append(i)
            elif name.startswith('AU'):
                au_indices.append(i)
        
        return pose_indices, gaze_indices, au_indices
        
    def forward(self, x):
        """
        Apply feature group balancing
        
        Args:
            x: Input features [batch_size, seq_length, input_size]
            
        Returns:
            Balanced features with same shape
        """
        batch_size, seq_length, input_size = x.shape
        
        # Create a copy to modify
        balanced_x = x.clone()
        
        # Apply fixed scaling to pose features to reduce their impact
        if self.pose_indices:
            balanced_x[:, :, self.pose_indices] *= self.pose_scaling_factor
        
        # Apply learned feature group weights (softmax normalized)
        weights = F.softmax(self.feature_group_weights, dim=0)
        
        # Apply weights to each feature group
        if self.pose_indices:
            balanced_x[:, :, self.pose_indices] *= weights[0]
        if self.gaze_indices:
            balanced_x[:, :, self.gaze_indices] *= weights[1]
        if self.au_indices:
            balanced_x[:, :, self.au_indices] *= weights[2]
            
        return balanced_x
    
    def get_regularization_loss(self):
        """
        Get L1 regularization loss for pose feature weights
        to encourage the model to rely less on pose features
        """
        weights = F.softmax(self.feature_group_weights, dim=0)
        return weights[0]  # L1 penalty on pose weight only
    
    def get_group_weights(self):
        """Return the current feature group weights (softmax normalized)"""
        return F.softmax(self.feature_group_weights, dim=0).detach()  # Detach tensor before returning

class DepBinaryClassifier(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=3, dropout=0.5, seq_length=150,
                 feature_names=None, include_pose=True, pose_scaling_factor=0.5):
        """
        Binary classification model for depression detection with simplified CNN architecture
        and feature group balancing.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): CNN hidden state size
            num_layers (int): Number of CNN layers
            dropout (float): Dropout probability
            seq_length (int): Length of input sequences
            feature_names (list): Names of input features for interpretability
            include_pose (bool): Whether pose features are included
            pose_scaling_factor (float): Scaling factor to reduce pose feature impact (0-1)
        """
        super(DepBinaryClassifier, self).__init__()

        # Store feature names for interpretability
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(input_size)]
        self.input_size = input_size
        self.include_pose = include_pose
        
        # Create feature group indices for feature importance analysis
        self.pose_indices, self.gaze_indices, self.au_indices = self._create_feature_group_indices()
        
        # Feature group balancer to handle pose feature dominance
        self.feature_balancer = FeatureGroupBalancer(
            feature_names=self.feature_names,
            pose_scaling_factor=pose_scaling_factor
        )
        
        # Single encoder for all features (simplified architecture)
        self.encoder = SimpleCNNEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            seq_length=seq_length,
            dropout=dropout,
            num_conv_layers=num_layers
        )
        
        # Final feature processing
        self.feature_processing = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer for binary classification
        self.fc = nn.Linear(hidden_size, 1)
        
        # For storing feature importance information
        self.feature_importances = {}
        self.last_attention_weights = None  # Kept for compatibility
    
    def _create_feature_group_indices(self):
        """Create indices for each feature group based on feature names"""
        pose_indices = []
        gaze_indices = []
        au_indices = []
        
        for i, name in enumerate(self.feature_names):
            if name.startswith('pose_') and self.include_pose:
                pose_indices.append(i)
            elif name.startswith('gaze_'):
                gaze_indices.append(i)
            elif name.startswith('AU'):
                au_indices.append(i)
        
        return pose_indices, gaze_indices, au_indices
    
    def forward(self, x, return_intermediates=False):
        """
        Forward pass with feature group balancing
        
        Args:
            x (torch.Tensor): Input features [batch, seq_length, input_size]
            return_intermediates (bool): Whether to return intermediate activations
            
        Returns:
            torch.Tensor or tuple: Binary classification logits, or tuple with intermediates
        """
        # Store intermediates if requested
        intermediates = {'input': x} if return_intermediates else {}
        
        # Apply feature group balancing
        balanced_x = self.feature_balancer(x)
        
        if return_intermediates:
            intermediates['balanced_input'] = balanced_x
        
        # Process through CNN encoder
        encoded_features = self.encoder(balanced_x)
        
        if return_intermediates:
            intermediates['encoded_features'] = encoded_features
        
        # Pool sequence to get a single vector per example
        pooled = self.encoder.pool_sequence(encoded_features)
        
        if return_intermediates:
            intermediates['pooled'] = pooled
        
        # Final processing
        features = self.feature_processing(pooled)
        
        # Output layer
        logits = self.fc(features)
        
        if return_intermediates:
            intermediates['final_features'] = features
            return logits, intermediates
        
        return logits
    
    def get_regularization_loss(self, reg_strength=0.01):
        """Get regularization loss to discourage over-reliance on pose features"""
        return reg_strength * self.feature_balancer.get_regularization_loss()
    
    def get_feature_group_weights(self):
        """Return the current feature group weights"""
        return self.feature_balancer.get_group_weights()
    
    def predict_proba(self, x):
        """Get probability predictions by applying sigmoid to logits"""
        logits = self(x)
        return torch.sigmoid(logits)
    
    def calculate_metrics(self, logits, targets, threshold=0.0):
        """Calculate binary classification metrics"""
        # Convert logits to binary predictions using threshold
        binary_preds = (logits > threshold).float()
        
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
        """Returns empty list since we don't use attention in the simplified model"""
        return []
    
    # Keep the existing feature importance methods for compatibility
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
        has_highlighted = False
        
        for i, feature in enumerate(feature_names):
            for au in highlighted_aus:
                if au in feature:
                    bars[i].set_color('red')
                    has_highlighted = True
                    break
        
        # Add legend to explain colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Clinically Significant AUs'),
            Patch(facecolor='cornflowerblue', label='Other Features')
        ]
        
        # Only show legend if we have at least one highlighted AU
        if has_highlighted:
            plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def visualize_attention(self, layer_idx=0, save_path=None):
        """
        Visualize attention weights
        
        Args:
            layer_idx (int): Index of the layer to visualize
            save_path (str): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Attention visualization
        """
        attention_weights = self.get_attention_weights()
        
        if not attention_weights:
            raise ValueError("No attention weights available. Run forward pass first.")
            
        if layer_idx >= len(attention_weights):
            raise ValueError(f"Layer index {layer_idx} out of range")
            
        # Extract weights from specified layer
        layer_name = attention_weights[layer_idx]['layer']
        weights = attention_weights[layer_idx]['weights']
        
        # Visualize attention weights
        plt.figure(figsize=(10, 6))
        plt.imshow(weights.cpu().numpy().squeeze(), aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.title(f'Attention Weights: {layer_name}')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Sequence Position')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
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
    
    def visualize_feature_group_weights(self, save_path=None):
        """Visualize current feature group weights"""
        weights = self.get_feature_group_weights().detach().cpu().numpy()
        
        # Create labels based on which groups exist
        labels = []
        values = []
        
        if self.pose_indices:
            labels.append('Pose Features')
            values.append(weights[0])
        if self.gaze_indices:
            labels.append('Gaze Features')
            values.append(weights[1])
        if self.au_indices:
            labels.append('Action Units')
            values.append(weights[2])
        
        # Create visualization
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=['lightcoral', 'lightblue', 'lightgreen'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylabel('Weight')
        plt.title('Feature Group Weights')
        plt.ylim(0, 1.0)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return plt.gcf()
