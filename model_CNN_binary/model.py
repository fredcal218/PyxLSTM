import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureGroupCNN(nn.Module):
    """CNN encoder for a specific group of features"""
    def __init__(self, input_size, hidden_size, seq_length, dropout=0.5, num_conv_layers=3):
        super(FeatureGroupCNN, self).__init__()
        
        # Initial projection to match hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # Create a stack of 1D convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            # Calculate proper padding to maintain sequence length with dilated convolutions
            dilation = 2**i
            kernel_size = 3
            padding = (kernel_size - 1) * dilation // 2
            
            # Each layer: conv -> batchnorm -> relu -> dropout
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=padding,  # Updated to scale with dilation
                    dilation=dilation
                ),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_layer)
        
        # Global attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
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
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x_conv = conv_layer(x_conv) + x_conv  # Residual connection
            
        # Transpose back to [batch, seq_len, hidden_size]
        x_out = x_conv.transpose(1, 2)
        
        return x_out
        
    def apply_attention(self, x):
        """Apply attention pooling to get sequence representation
        Args:
            x: Tensor [batch_size, seq_length, hidden_size]
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Calculate attention scores
        attn_scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)  # [batch, seq_len, 1]
        
        # Store for visualization
        self.last_attn_weights = attn_weights.detach()
        
        # Apply attention weights
        attended = torch.sum(x * attn_weights, dim=1)  # [batch, hidden_size]
        
        return attended

class DepBinaryClassifier(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=3, dropout=0.5, seq_length=150,
                 feature_names=None, include_pose=True):
        """
        Binary classification model for E-DAIC depression detection with interpretability
        using 1D CNNs instead of LSTM/Transformer.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): CNN hidden state size
            num_layers (int): Number of CNN layers
            dropout (float): Dropout probability
            seq_length (int): Length of input sequences
            feature_names (list): Names of input features for interpretability
            include_pose (bool): Whether pose features are included
        """
        super(DepBinaryClassifier, self).__init__()

        # Store feature names for interpretability
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(input_size)]
        self.input_size = input_size
        self.include_pose = include_pose
        
        # Create feature group indices
        self.pose_indices, self.gaze_indices, self.au_indices = self._create_feature_group_indices()
        
        # If pose features are included, create separate encoders
        if include_pose and len(self.pose_indices) > 0:
            # Create separate encoders for each feature group
            self.pose_encoder = FeatureGroupCNN(
                input_size=len(self.pose_indices),
                hidden_size=hidden_size // 2,  # Smaller dimension for pose features
                seq_length=seq_length,
                dropout=dropout * 1.5,  # Higher dropout for pose features
                num_conv_layers=num_layers
            )
            
            self.au_encoder = FeatureGroupCNN(
                input_size=len(self.au_indices) + len(self.gaze_indices),  # Combine AUs and gaze
                hidden_size=hidden_size // 2 + hidden_size // 4,  # Larger dimension for AU features
                seq_length=seq_length,
                dropout=dropout,
                num_conv_layers=num_layers
            )
            
            # Merger for feature group encodings
            merged_size = hidden_size + hidden_size // 4
            self.feature_merger = nn.Linear(merged_size, hidden_size)
            self.merger_dropout = nn.Dropout(dropout)
            
            # Attention layer for combining feature groups
            self.feature_attention = nn.Linear(hidden_size // 2, 2)  # Match the pose encoder dimension
        else:
            # If no pose features, use a single encoder for AUs and gaze
            self.au_gaze_encoder = FeatureGroupCNN(
                input_size=len(self.au_indices) + len(self.gaze_indices),
                hidden_size=hidden_size,
                seq_length=seq_length,
                dropout=dropout,
                num_conv_layers=num_layers
            )
        
        # Final sequence model: a stack of 1D CNN layers with global pooling
        self.feature_pooling = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(), 
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Output layer for binary classification
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        # For storing feature importance information
        self.feature_importances = {}
        self.last_attention_weights = None
    
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
    
    def _split_features(self, x):
        """Split input features into feature groups"""
        # Create empty tensors for each group
        batch_size, seq_len, _ = x.shape
        
        if self.include_pose and len(self.pose_indices) > 0:
            pose_features = torch.zeros(batch_size, seq_len, len(self.pose_indices), device=x.device)
            au_gaze_features = torch.zeros(batch_size, seq_len, len(self.au_indices) + len(self.gaze_indices), device=x.device)
            
            # Fill with appropriate features
            for i, idx in enumerate(self.pose_indices):
                pose_features[:, :, i] = x[:, :, idx]
            
            au_gaze_idx = 0
            for idx in self.gaze_indices + self.au_indices:
                au_gaze_features[:, :, au_gaze_idx] = x[:, :, idx]
                au_gaze_idx += 1
            
            return pose_features, au_gaze_features
        else:
            # If no pose features, just return AU and gaze features
            au_gaze_features = torch.zeros(batch_size, seq_len, len(self.au_indices) + len(self.gaze_indices), device=x.device)
            
            au_gaze_idx = 0
            for idx in self.gaze_indices + self.au_indices:
                au_gaze_features[:, :, au_gaze_idx] = x[:, :, idx]
                au_gaze_idx += 1
            
            return None, au_gaze_features
    
    def forward(self, x, return_intermediates=False):
        """
        Forward pass with separate pathways for feature groups if pose is included
        
        Args:
            x (torch.Tensor): Input features [batch, seq_length, input_size]
            return_intermediates (bool): Whether to return intermediate activations
            
        Returns:
            torch.Tensor or tuple: Binary classification logits, or tuple with intermediates
        """
        # x shape: [batch, seq_length, input_size]
        batch_size, seq_len, _ = x.shape
        
        # Split features into groups
        pose_features, au_gaze_features = self._split_features(x)
        
        intermediates = {'input': x, 'au_gaze_features': au_gaze_features}
        
        if self.include_pose and pose_features is not None:
            # Process each feature group through its encoder
            pose_encoded = self.pose_encoder(pose_features)
            au_gaze_encoded = self.au_encoder(au_gaze_features)
            
            intermediates['pose_features'] = pose_features
            intermediates['pose_encoded'] = pose_encoded
            intermediates['au_gaze_encoded'] = au_gaze_encoded
            
            # Apply attention to pose features
            pose_attended = self.pose_encoder.apply_attention(pose_encoded)
            
            # Apply attention to AU/gaze features
            au_gaze_attended = self.au_encoder.apply_attention(au_gaze_encoded)
            
            # Apply feature attention to combine groups
            attention_weights = F.softmax(self.feature_attention(pose_attended), dim=-1)
            self.last_group_attention = attention_weights.detach()
            
            intermediates['group_attention'] = attention_weights
            
            # Merge feature group encodings
            merged = torch.cat([pose_encoded, au_gaze_encoded], dim=2)
            merged = self.merger_dropout(self.feature_merger(merged))
            
            intermediates['merged'] = merged
            
            # Apply global attention pooling
            attn_scores = self.global_attention(merged).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)
            self.last_attention_weights = attn_weights.detach()
            
            out_seq = merged
            pooled = torch.sum(out_seq * attn_weights, dim=1)
            
        else:
            # If no pose features, process through the AU/gaze encoder
            out_seq = self.au_gaze_encoder(au_gaze_features)
            
            # Apply global attention pooling
            attn_scores = self.global_attention(out_seq).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)
            self.last_attention_weights = attn_weights.detach()
            
            pooled = torch.sum(out_seq * attn_weights, dim=1)
        
        intermediates['sequence_out'] = out_seq
        intermediates['attention_weights'] = attn_weights
        
        # Apply feature pooling and final classification
        out_features = self.feature_pooling(pooled)
        out_features = self.dropout(out_features)
        logits = self.fc(out_features)
        
        intermediates['final_out'] = out_features
        
        # Return raw logits (no sigmoid) for use with BCEWithLogitsLoss
        if return_intermediates:
            return logits, intermediates
        
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
        Extract attention weights from the model
        
        Returns:
            list: Attention weights from each layer
        """
        attention_weights = []
        
        # Extract feature group attention if available
        if hasattr(self, 'last_group_attention'):
            attention_weights.append({
                'layer': 'feature_group_attention',
                'weights': self.last_group_attention
            })
        
        # Extract global attention weights if available
        if hasattr(self, 'last_attention_weights'):
            attention_weights.append({
                'layer': 'global_attention',
                'weights': self.last_attention_weights
            })
        
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
