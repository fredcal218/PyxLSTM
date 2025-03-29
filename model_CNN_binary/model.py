import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important frames in the sequence.
    
    This module computes attention weights for each time step in the sequence,
    allowing the model to focus on the most relevant frames for depression detection.
    """
    def __init__(self, hidden_size, attention_size=64):
        super(TemporalAttention, self).__init__()
        
        # Attention projection layers
        self.query = nn.Linear(hidden_size, attention_size)
        self.key = nn.Linear(hidden_size, attention_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Scoring layer
        self.score = nn.Linear(attention_size, 1)
        
        # Save attention weights for visualization
        self.last_attention_weights = None
        
    def forward(self, x):
        """
        Apply temporal attention to focus on important frames
        
        Args:
            x: Input tensor [batch_size, seq_length, hidden_size]
            
        Returns:
            Attended features [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Project to query and key spaces
        q = torch.tanh(self.query(x))  # [batch, seq_len, attn_size]
        k = torch.tanh(self.key(x))    # [batch, seq_len, attn_size]
        
        # Compute attention scores
        scores = self.score(q + k)     # [batch, seq_len, 1]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=1)  # [batch, seq_len, 1]
        
        # Save attention weights for visualization
        self.last_attention_weights = attn_weights.detach()
        
        # Apply attention weights to values
        v = self.value(x)  # [batch, seq_len, hidden_size]
        
        # Weighted sum
        context = torch.bmm(attn_weights.transpose(1, 2), v)  # [batch, 1, hidden_size]
        attended_features = context.squeeze(1)  # [batch, hidden_size]
        
        return attended_features
    
    def get_attention_weights(self):
        """Return the last computed attention weights for visualization"""
        if self.last_attention_weights is None:
            return None
        return self.last_attention_weights

class SimpleCNNEncoder(nn.Module):
    """Simplified CNN encoder for all features combined with temporal attention"""
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
        
        # Add temporal attention for sequence pooling
        self.temporal_attention = TemporalAttention(hidden_size)
        
        # Keep global pool as a fallback option
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Store attention weights
        self.last_attention_weights = None
        
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
        """Pool sequence into a single vector using temporal attention
        Args:
            x: Tensor [batch_size, seq_length, hidden_size]
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Apply temporal attention to get weighted feature representation
        attended_features = self.temporal_attention(x)
        
        # Store attention weights for visualization
        self.last_attention_weights = self.temporal_attention.get_attention_weights()
        
        return attended_features

    def get_attention_weights(self):
        """Return the temporal attention weights"""
        return self.last_attention_weights

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

class ClinicalFeatureBalancer(nn.Module):
    """
    Enhanced feature balancer that incorporates clinical knowledge about
    depression-relevant facial features and movement patterns.
    
    This module applies feature-specific scaling based on clinical relevance
    in depression detection literature.
    """
    def __init__(self, feature_names, pose_scaling_factor=0.1):
        super(ClinicalFeatureBalancer, self).__init__()
        
        # Categorize features by type and clinical relevance
        self.feature_indices = self._categorize_features(feature_names)
        
        # Store original feature names
        self.feature_names = feature_names
        
        # Set feature group scaling factors (fixed)
        self.pose_scaling_factor = pose_scaling_factor
        
        # Create learnable weights for feature groups, initialized with clinical knowledge
        self.feature_group_weights = nn.Parameter(
            self._initialize_clinical_weights()
        )
        
        # Document clinical knowledge for interpretability
        self.clinical_knowledge = {
            # Depression-relevant AUs and their clinical significance
            'depression_aus': {
                'AU01': 'Inner Brow Raiser - associated with worry/anxiety',
                'AU04': 'Brow Lowerer - associated with sadness/worry',
                'AU05': 'Upper Lid Raiser - associated with fear/worry',
                'AU07': 'Lid Tightener - eye narrowing in negative emotions',
                'AU15': 'Lip Corner Depressor - associated with sadness',
                'AU17': 'Chin Raiser - associated with contemplation/doubt',
                'AU20': 'Lip Stretcher - associated with fear/tension',
                'AU26': 'Jaw Drop - associated with decreased expressivity'
            },
            # Social engagement AUs that typically decrease in depression
            'social_aus': {
                'AU06': 'Cheek Raiser - genuine smile component (Duchenne)',
                'AU12': 'Lip Corner Puller - smile component',
                'AU14': 'Dimpler - associated with positive emotions'
            },
            # Head pose features relevant to depression
            'depression_pose': {
                'pose_Rz': 'Head yaw (turning) - indicates reduced engagement',
                'pose_Rx': 'Head pitch (nodding) - when reduced indicates decreased expressivity',
                'pose_Tx_velocity': 'Reduced head movement velocity - psychomotor retardation'
            }
        }
    
    def _categorize_features(self, feature_names):
        """Categorize features by type and clinical relevance"""
        # Define clinically significant feature groups
        depression_relevant_aus = ['AU01', 'AU04', 'AU05', 'AU07', 'AU15', 'AU17', 'AU20', 'AU26']
        social_engagement_aus = ['AU06', 'AU12', 'AU14']
        depression_relevant_pose = ['pose_Rz', 'pose_Rx', 'pose_Tx_velocity', 'pose_Ty_velocity']
        
        # Initialize feature category indices
        feature_indices = {
            'depression_aus': [],
            'social_aus': [],
            'depression_pose': [],
            'other_pose': [],
            'gaze': [],
            'other_aus': []
        }
        
        # Categorize each feature
        for i, name in enumerate(feature_names):
            # Check AUs
            if any(au in name for au in depression_relevant_aus):
                feature_indices['depression_aus'].append(i)
            elif any(au in name for au in social_engagement_aus):
                feature_indices['social_aus'].append(i)
            elif name.startswith('AU'):
                feature_indices['other_aus'].append(i)
            # Check pose
            elif any(pose in name for pose in depression_relevant_pose):
                feature_indices['depression_pose'].append(i)
            elif name.startswith('pose_'):
                feature_indices['other_pose'].append(i)
            # Check gaze
            elif name.startswith('gaze_'):
                feature_indices['gaze'].append(i)
        
        return feature_indices
    
    def _initialize_clinical_weights(self):
        """Initialize learnable weights with clinical knowledge"""
        # Standard initial weight (neutral)
        standard_weight = 1.0
        
        # Enhanced weight for depression-relevant features
        depression_au_weight = 1.5
        social_au_weight = 1.2
        depression_pose_weight = 1.3
        other_pose_weight = 0.8
        gaze_weight = 1.0
        other_au_weight = 0.9
        
        # Create initial weights tensor [depression_aus, social_aus, depression_pose, 
        #                                other_pose, gaze, other_aus]
        weights = torch.tensor([
            depression_au_weight,
            social_au_weight,
            depression_pose_weight,
            other_pose_weight,
            gaze_weight,
            other_au_weight
        ])
        
        return weights
    
    def forward(self, x):
        """
        Apply clinically-informed feature balancing
        
        Args:
            x: Input features [batch_size, seq_length, input_size]
            
        Returns:
            Balanced features with same shape
        """
        batch_size, seq_length, input_size = x.shape
        
        # Create a copy to modify
        balanced_x = x.clone()
        
        # Normalize weights with softmax to maintain relative scaling
        weights = F.softmax(self.feature_group_weights, dim=0)
        
        # Apply weights to each feature category
        for i, (category, indices) in enumerate(self.feature_indices.items()):
            if indices:  # Only if we have features in this category
                if category in ['depression_pose', 'other_pose']:
                    # Apply additional pose scaling factor to reduce overall dominance
                    balanced_x[:, :, indices] *= self.pose_scaling_factor
                
                # Apply learned weights for this feature category
                balanced_x[:, :, indices] *= weights[i]
        
        return balanced_x
    
    def get_regularization_loss(self, clinical_strength=0.5):
        """
        Calculate regularization loss to encourage clinically aligned weighting
        
        Args:
            clinical_strength: Strength of clinical prior regularization
            
        Returns:
            Regularization loss value
        """
        # Get normalized weights
        weights = F.softmax(self.feature_group_weights, dim=0)
        
        # We want depression_aus and depression_pose to have higher weights
        # than other_aus and other_pose respectively
        depression_au_weight = weights[0]  # depression_aus
        social_au_weight = weights[1]      # social_aus
        depression_pose_weight = weights[2]  # depression_pose
        other_pose_weight = weights[3]     # other_pose
        
        # Clinical regularization: encourage depression_aus > other_aus
        # and depression_pose > other_pose
        clinical_reg = torch.relu(other_pose_weight - depression_pose_weight) + \
                       torch.relu(other_pose_weight - 0.7 * social_au_weight)
        
        return clinical_strength * clinical_reg
    
    def get_category_weights(self):
        """Return the current feature category weights (softmax normalized)"""
        return F.softmax(self.feature_group_weights, dim=0).detach()
    
    def get_category_names(self):
        """Return the category names in the same order as weights"""
        return list(self.feature_indices.keys())
    
    def visualize_clinical_weights(self, save_path=None):
        """Visualize the clinical feature weights"""
        weights = self.get_category_weights().cpu().numpy()
        categories = self.get_category_names()
        
        # Check if any categories have features (non-empty)
        active_categories = []
        active_weights = []
        colors = []
        
        # Define colors for clinical relevance
        category_colors = {
            'depression_aus': 'darkred',
            'social_aus': 'lightcoral',
            'depression_pose': 'firebrick',
            'other_pose': 'lightblue',
            'gaze': 'cornflowerblue',
            'other_aus': 'darkblue'
        }
        
        # Only include categories that have features
        for i, category in enumerate(categories):
            if len(self.feature_indices[category]) > 0:
                active_categories.append(category)
                active_weights.append(weights[i])
                colors.append(category_colors.get(category, 'gray'))
        
        if not active_categories:
            raise ValueError("No feature categories are active with current modality configuration")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        bars = plt.bar(active_categories, active_weights, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylabel('Weight')
        plt.title('Clinical Feature Category Weights')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(active_weights) * 1.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return plt.gcf()

class DepBinaryClassifier(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=3, dropout=0.5, seq_length=150,
                 feature_names=None, include_pose=True, pose_scaling_factor=0.5, use_clinical_knowledge=True):
        """
        Binary classification model for depression detection with CNN architecture,
        temporal attention, and clinical knowledge integration.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): CNN hidden state size
            num_layers (int): Number of CNN layers
            dropout (float): Dropout probability
            seq_length (int): Length of input sequences
            feature_names (list): Names of input features for interpretability
            include_pose (bool): Whether pose features are included
            pose_scaling_factor (float): Scaling factor to reduce pose feature impact (0-1)
            use_clinical_knowledge (bool): Whether to use enhanced clinical knowledge integration
        """
        super(DepBinaryClassifier, self).__init__()

        # Store feature names for interpretability
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(input_size)]
        self.input_size = input_size
        self.include_pose = include_pose
        self.use_clinical_knowledge = use_clinical_knowledge
        
        # Store modality flags (inferred from feature names)
        self.include_gaze = any(name.startswith('gaze_') for name in self.feature_names)
        self.include_au = any(name.startswith('AU') for name in self.feature_names)
        
        # Create feature group indices for feature importance analysis
        self.pose_indices, self.gaze_indices, self.au_indices = self._create_feature_group_indices()
        
        # Feature balancer - choose between standard or clinical knowledge-enhanced
        if use_clinical_knowledge:
            self.feature_balancer = ClinicalFeatureBalancer(
                feature_names=self.feature_names,
                pose_scaling_factor=pose_scaling_factor
            )
        else:
            # Use the original feature balancer
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
        # Check if we're using clinical feature balancer
        if hasattr(self.feature_balancer, 'get_category_weights'):
            # For single modality cases, we need to handle this differently
            # Only return weights for categories that have features
            weights = self.feature_balancer.get_category_weights()
            
            # If we have a standard feature balancer fallback for visualization
            if not self.include_pose and not self.include_gaze and self.include_au:
                # AU only mode - create a simplified weight tensor
                return torch.tensor([0.0, 0.0, 1.0]).to(weights.device)
            elif self.include_pose and not self.include_gaze and not self.include_au:
                # Pose only mode
                return torch.tensor([1.0, 0.0, 0.0]).to(weights.device)
            elif not self.include_pose and self.include_gaze and not self.include_au:
                # Gaze only mode
                return torch.tensor([0.0, 1.0, 0.0]).to(weights.device)
            
            return weights
        else:
            # Standard feature group balancer
            return self.feature_balancer.get_group_weights()
    
    def get_probabilities(self, x):
        """Apply sigmoid to logits"""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def calculate_metrics(self, logits, targets, threshold=0.5):
        """Calculate binary classification metrics"""
        binary_preds = (logits >= threshold).float()
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
        """Returns temporal attention weights if available"""
        if hasattr(self.encoder, 'get_attention_weights'):
            weights = self.encoder.get_attention_weights()
            if weights is not None:
                return [{'layer': 'temporal_attention', 'weights': weights}]
        return []
    
    def gradient_feature_importance(self, x, target_class=1):
        """
        Calculate feature importance using gradient-based attribution
        
        Args:
            x (torch.Tensor): Input features [batch, seq_len, feature_dim]
            target_class (int): Target class (1=depressed, 0=not depressed)
            
        Returns:
            dict: Importance scores for each feature
        """
        self.eval()
        x_input = x.clone().detach().requires_grad_(True)
        logits = self(x_input)
        
        # Select target based on class - use logits directly
        if target_class == 1:
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
        
        # Get most recent feature importance
        if self.feature_importances is None or 'global' not in self.feature_importances:
            raise ValueError("No feature importance data available. Run gradient_feature_importance() first.")
        
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
        # Check if we're using clinical feature balancer
        if hasattr(self.feature_balancer, 'get_category_weights'):
            # Using clinical feature balancer
            # Use the balancer's own visualization method
            if hasattr(self.feature_balancer, 'visualize_clinical_weights'):
                try:
                    return self.feature_balancer.visualize_clinical_weights(save_path=save_path)
                except ValueError as e:
                    print(f"Cannot visualize clinical weights: {e}")
                    print("Falling back to standard feature group visualization")
        
        # Using standard feature group balancer or fallback
        weights = self.get_feature_group_weights().detach().cpu().numpy()
        
        # Create labels based on which groups exist
        labels = []
        values = []
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        bar_colors = []
        
        # Only include feature groups that exist
        if self.pose_indices:
            labels.append('Pose Features')
            values.append(weights[0])
            bar_colors.append(colors[0])
        
        if self.gaze_indices:
            labels.append('Gaze Features')
            values.append(weights[1])
            bar_colors.append(colors[1])
        
        if self.au_indices:
            labels.append('Action Units')
            values.append(weights[2])
            bar_colors.append(colors[2])
        
        if not labels:
            print("Warning: No feature groups detected for visualization")
            return None
        
        # Create visualization
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=bar_colors)
        
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
    
    def visualize_temporal_attention(self, x, save_path=None):
        """
        Visualize temporal attention weights across sequence
        
        Args:
            x: Input features [batch_size, seq_length, feature_dim]
            save_path: Path to save the visualization
            
        Returns:
            matplotlib figure
        """
        # Get temporal attention weights by running forward pass
        with torch.no_grad():
            attention_weights = self.encoder.get_attention_weights()
            if attention_weights is None:
                # Forward pass to compute attention weights
                _ = self(x)
                attention_weights = self.encoder.get_attention_weights()
        
        if attention_weights is None:
            raise ValueError("No attention weights available. Run forward pass first.")
        
        # Convert to numpy for visualization
        attention = attention_weights.cpu().numpy().squeeze()  # [seq_len, 1]
        
        # Create visualization
        plt.figure(figsize=(12, 4))
        
        # Plot attention weights
        plt.subplot(1, 1, 1)
        plt.plot(attention, 'b-', linewidth=2)
        plt.fill_between(range(len(attention)), 0, attention.flatten(), alpha=0.2, color='blue')
        
        # Mark important frames
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(attention.flatten(), height=0.01, distance=5)
        if len(peaks) > 0:
            plt.plot(peaks, attention.flatten()[peaks], "ro", label='Important Frames')
            plt.legend(loc='upper right')
        
        plt.title('Temporal Attention Weights')
        plt.xlabel('Frame Index')
        plt.ylabel('Attention Weight')
        plt.grid(True, alpha=0.3)
        
        # Highlight regions with high attention
        threshold = np.mean(attention) + np.std(attention)
        high_attention = attention.flatten() > threshold
        if np.any(high_attention):
            plt.axhline(y=threshold, color='r', linestyle='--', 
                      label=f'Significance Threshold ({threshold:.3f})')
            plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()