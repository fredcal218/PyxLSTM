import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now this import will work
from model_binary.xlstm import(  
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class TargetedFeatureDropout(nn.Module):
    """
    Applies a higher dropout rate to specific features during training.
    Unlike standard dropout, this applies a fixed dropout pattern across the sequence
    for each batch, encouraging the model to rely less on the targeted features.
    """
    def __init__(self, dropout_rate=0.5):
        super(TargetedFeatureDropout, self).__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, x, training=True):
        """
        Apply dropout to input features
        
        Args:
            x: Input features tensor [batch_size, seq_length, feature_dim]
            training: Whether in training mode (apply dropout) or not
            
        Returns:
            Features with dropout applied
        """
        if not training or self.dropout_rate == 0.0:
            return x
            
        # Create a binary mask that's the same across the sequence dimension
        # This makes certain pose features consistently unavailable for the entire sequence
        # Shape: [batch_size, 1, feature_dim]
        batch_size, seq_len, feature_dim = x.shape
        
        mask = torch.bernoulli(
            torch.ones(batch_size, 1, feature_dim, device=x.device) * (1.0 - self.dropout_rate)
        ).expand(-1, seq_len, -1) / (1.0 - self.dropout_rate)  # Scale to maintain expectation
            
        return x * mask

class CNNEncoder(nn.Module):
    """CNN encoder for facial expression features (AUs and gaze)"""
    def __init__(self, input_size, hidden_size, seq_length, dropout=0.5, num_conv_layers=3):
        super(CNNEncoder, self).__init__()
        
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
                    padding=padding,
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

class xLSTMEncoder(nn.Module):
    """xLSTM encoder for temporal pose features"""
    def __init__(self, input_size, hidden_size, seq_length, dropout=0.5, num_blocks=1):
        super(xLSTMEncoder, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # xLSTM configuration
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
            num_blocks=num_blocks,
            embedding_dim=hidden_size,
            slstm_at=[],  # Use mLSTM for simplicity
            dropout=dropout,
            bias=True,
        )
        
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.xlstm_stack.reset_parameters()
    
    def forward(self, x):
        """Forward pass for xLSTM encoder"""
        # Project and apply dropout
        x_proj = self.input_dropout(self.input_proj(x))
        
        # Process through xLSTM stack
        out_seq = self.xlstm_stack(x_proj)
        
        # Return full sequence output
        return out_seq

class AudioEncoder(nn.Module):
    """xLSTM encoder for audio features"""
    def __init__(self, input_size, hidden_size, seq_length, dropout=0.5, num_blocks=1):
        super(AudioEncoder, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # xLSTM configuration
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
            num_blocks=num_blocks,
            embedding_dim=hidden_size,
            slstm_at=[],  # Use mLSTM for simplicity
            dropout=dropout,
            bias=True,
        )
        
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.xlstm_stack.reset_parameters()
        
        # Add global attention mechanism for sequence pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass for xLSTM encoder"""
        # Project and apply dropout
        x_proj = self.input_dropout(self.input_proj(x))
        
        # Process through xLSTM stack
        out_seq = self.xlstm_stack(x_proj)
        
        # Return full sequence output
        return out_seq
    
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

class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer for effective information exchange between modalities.
    Enables each modality to attend to others, creating richer representations.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(CrossModalTransformer, self).__init__()
        self.hidden_size = hidden_size
        
        # Multi-head attention mechanisms
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        Args:
            query: Features from one modality [batch, seq_len, hidden_size]
            key_value: Features from another modality [batch, seq_len, hidden_size]
            
        Returns:
            Enhanced query features with cross-modal context
        """
        # Cross-attention: query attends to key_value
        attn_output, attn_weights = self.cross_attention(
            query=self.norm1(query),
            key=self.norm1(key_value),
            value=self.norm1(key_value)
        )
        
        # Store attention weights for visualization
        self.last_attn_weights = attn_weights.detach()
        
        # Residual connection
        query = query + self.dropout(attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(self.norm2(query))
        
        # Final residual connection
        output = query + self.dropout(ffn_output)
        
        return output

class HybridDepBinaryClassifier(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=2, dropout=0.5, pose_dropout=0.4, 
                 audio_dropout=0.3, seq_length=150, feature_names=None, audio_feature_names=None,
                 include_pose=True, include_audio=True, use_hybrid_pathways=True, 
                 fusion_type="attention"):
        """
        Hybrid CNN-xLSTM-Audio model for depression detection
        
        Args:
            input_size (int): Size of visual input features
            hidden_size (int): Hidden state size
            num_layers (int): Number of layers for encoders
            dropout (float): Dropout probability
            pose_dropout (float): Higher dropout specifically for pose features
            audio_dropout (float): Dropout for audio features
            seq_length (int): Length of input sequences
            feature_names (list): Names of visual input features for interpretability
            audio_feature_names (list): Names of audio features for interpretability
            include_pose (bool): Whether pose features are included
            include_audio (bool): Whether audio features are included
            use_hybrid_pathways (bool): Whether to use multiple pathways regardless of feature types
            fusion_type (str): Type of fusion to use: "attention", "cross_modal"
        """
        super(HybridDepBinaryClassifier, self).__init__()
        
        # Store feature names and configuration
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(input_size)]
        self.audio_feature_names = audio_feature_names if audio_feature_names else []
        self.input_size = input_size
        self.audio_input_size = len(self.audio_feature_names) if self.audio_feature_names else 0
        self.include_pose = include_pose
        self.include_audio = include_audio
        self.pose_dropout = pose_dropout
        self.audio_dropout = audio_dropout
        self.use_hybrid_pathways = use_hybrid_pathways
        self.fusion_type = fusion_type
        
        # Create feature group indices
        self.pose_indices, self.gaze_indices, self.au_indices = self._create_feature_group_indices()
        
        # Initialize targeted dropout for pose and audio features
        self.pose_feature_dropout = TargetedFeatureDropout(dropout_rate=pose_dropout)
        self.audio_feature_dropout = TargetedFeatureDropout(dropout_rate=audio_dropout)
        
        # Create pathways based on enabled features
        if self.use_hybrid_pathways:
            # Determine input sizes based on available features
            if include_pose and len(self.pose_indices) > 0:
                # When pose is included, use pose for xLSTM
                pose_input_size = len(self.pose_indices)
                # Use AU/gaze for CNN
                cnn_input_size = len(self.au_indices) + len(self.gaze_indices)
            else:
                # When pose is excluded, distribute AU/gaze features to both pathways
                total_vis_features = len(self.au_indices) + len(self.gaze_indices)
                pose_input_size = total_vis_features // 3  # 1/3 for pose pathway
                cnn_input_size = total_vis_features - pose_input_size  # Rest for CNN
            
            # Sizes for each encoder pathway (roughly proportional to their importance)
            # Modified: Calculate hidden sizes that are divisible by 4 (number of attention heads)
            # We'll use approximately 1/3 of total hidden size for each modality, 
            # but ensure divisibility by 4
            
            # Base hidden size divisible by 12 (so each 1/3 is divisible by 4)
            adjusted_hidden_size = hidden_size - (hidden_size % 12)
            
            # Allocate hidden sizes to be approximately 1/3 each but ensure divisibility by 4
            pose_hidden_size = adjusted_hidden_size // 3
            pose_hidden_size = pose_hidden_size - (pose_hidden_size % 4)  # Ensure divisible by 4
            
            au_hidden_size = adjusted_hidden_size // 3
            au_hidden_size = au_hidden_size - (au_hidden_size % 4)  # Ensure divisible by 4
            
            audio_hidden_size = adjusted_hidden_size // 3
            audio_hidden_size = audio_hidden_size - (audio_hidden_size % 4)  # Ensure divisible by 4
            
            # Print hidden sizes for debugging
            print(f"Adjusted pathway hidden sizes: Pose={pose_hidden_size}, AU/Gaze={au_hidden_size}, Audio={audio_hidden_size}")
            
            # Create pose xLSTM encoder
            self.pose_encoder = xLSTMEncoder(
                input_size=pose_input_size,
                hidden_size=pose_hidden_size,
                seq_length=seq_length,
                dropout=dropout,
                num_blocks=num_layers
            )
            
            # Create CNN encoder for facial expressions
            self.au_encoder = CNNEncoder(
                input_size=cnn_input_size,
                hidden_size=au_hidden_size,
                seq_length=seq_length,
                dropout=dropout,
                num_conv_layers=num_layers
            )
            
            # Create audio encoder (if audio features are included)
            if include_audio and self.audio_input_size > 0:
                self.audio_encoder = AudioEncoder(
                    input_size=self.audio_input_size,
                    hidden_size=audio_hidden_size,
                    seq_length=seq_length,
                    dropout=dropout,
                    num_blocks=num_layers
                )
            
            # FUSION MECHANISMS
            if self.fusion_type == "cross_modal":
                # Cross-modal transformer fusion
                if include_audio and self.audio_input_size > 0:
                    # Three-way cross-modal attention
                    self.cm_pose_au = CrossModalTransformer(hidden_size=pose_hidden_size, dropout=dropout)
                    self.cm_pose_audio = CrossModalTransformer(hidden_size=pose_hidden_size, dropout=dropout)
                    self.cm_au_pose = CrossModalTransformer(hidden_size=au_hidden_size, dropout=dropout)
                    self.cm_au_audio = CrossModalTransformer(hidden_size=au_hidden_size, dropout=dropout)
                    self.cm_audio_pose = CrossModalTransformer(hidden_size=audio_hidden_size, dropout=dropout)
                    self.cm_audio_au = CrossModalTransformer(hidden_size=audio_hidden_size, dropout=dropout)
                else:
                    # Two-way cross-modal attention
                    self.cm_pose_au = CrossModalTransformer(hidden_size=pose_hidden_size, dropout=dropout)
                    self.cm_au_pose = CrossModalTransformer(hidden_size=au_hidden_size, dropout=dropout)
                
                # Merger for final representation
                merged_size = pose_hidden_size + au_hidden_size
                if include_audio and self.audio_input_size > 0:
                    merged_size += audio_hidden_size
                
                self.feature_merger = nn.Linear(merged_size, hidden_size)
                self.merger_dropout = nn.Dropout(dropout)
            else:
                # Default to original attention-based fusion
                # Merger for all feature group encodings
                merged_size = pose_hidden_size + au_hidden_size
                if include_audio and self.audio_input_size > 0:
                    merged_size += audio_hidden_size
                
                self.feature_merger = nn.Linear(merged_size, hidden_size)
                self.merger_dropout = nn.Dropout(dropout)
                
                # BALANCED FUSION: Create separate projections for each pathway
                # Project all pathways to a common dimension for fair comparison
                self.pose_projection = nn.Linear(pose_hidden_size, hidden_size // 4)
                self.au_projection = nn.Linear(au_hidden_size, hidden_size // 4)
                
                if include_audio and self.audio_input_size > 0:
                    self.audio_projection = nn.Linear(audio_hidden_size, hidden_size // 4)
                    # Balanced fusion attention mechanism that uses all three pathways
                    self.feature_attention = nn.Sequential(
                        nn.Linear(hidden_size // 4 * 3, hidden_size // 4),  # Combined dimension from three pathways
                        nn.Tanh(),
                        nn.Linear(hidden_size // 4, 3)  # 3 outputs for the three pathways
                    )
                else:
                    # Two-way attention if audio is not included
                    self.feature_attention = nn.Sequential(
                        nn.Linear(hidden_size // 4 * 2, hidden_size // 4),  # Combined dimension from two pathways
                        nn.Tanh(),
                        nn.Linear(hidden_size // 4, 2)  # 2 outputs for the two pathways
                    )
        else:
            # If not using hybrid pathways, fall back to single encoder for all features
            combined_input_size = 0
            if include_pose:
                combined_input_size += len(self.pose_indices)
            combined_input_size += len(self.au_indices) + len(self.gaze_indices)
            
            self.au_gaze_encoder = CNNEncoder(
                input_size=combined_input_size,
                hidden_size=hidden_size,
                seq_length=seq_length,
                dropout=dropout,
                num_conv_layers=num_layers
            )
            
            # Separate audio encoder even in non-hybrid mode if audio is included
            if include_audio and self.audio_input_size > 0:
                self.audio_encoder = AudioEncoder(
                    input_size=self.audio_input_size,
                    hidden_size=hidden_size // 3,
                    seq_length=seq_length,
                    dropout=dropout,
                    num_blocks=num_layers
                )
                
                # Two-way merger for non-hybrid visual + audio
                self.feature_merger = nn.Linear(hidden_size + (hidden_size // 3), hidden_size)
                self.merger_dropout = nn.Dropout(dropout)
                
                # Two-way attention for visual + audio in non-hybrid mode
                self.visual_projection = nn.Linear(hidden_size, hidden_size // 4)
                self.audio_projection = nn.Linear(hidden_size // 3, hidden_size // 4)
                self.feature_attention = nn.Sequential(
                    nn.Linear(hidden_size // 4 * 2, hidden_size // 4),
                    nn.Tanh(),
                    nn.Linear(hidden_size // 4, 2)  # 2 outputs (visual + audio)
                )
        
        # Final processing layers
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
        self.last_modality_weights = None
    
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
    
    def _split_features(self, x, audio_features=None):
        """
        Split input features into feature groups for different pathways
        
        Args:
            x (torch.Tensor): Visual features [batch, seq_length, visual_features]
            audio_features (torch.Tensor): Audio features [batch, seq_length, audio_features]
            
        Returns:
            tuple: Feature tensors for each pathway
        """
        # Create empty tensors for each group
        batch_size, seq_len, _ = x.shape
        
        if self.use_hybrid_pathways:
            # When using hybrid pathways...
            if self.include_pose and len(self.pose_indices) > 0:
                # With pose: pose goes to xLSTM, AU/gaze to CNN
                pose_features = torch.zeros(batch_size, seq_len, len(self.pose_indices), device=x.device)
                au_gaze_features = torch.zeros(batch_size, seq_len, len(self.au_indices) + len(self.gaze_indices), device=x.device)
                
                # Fill with appropriate features
                for i, idx in enumerate(self.pose_indices):
                    pose_features[:, :, i] = x[:, :, idx]
                
                au_gaze_idx = 0
                for idx in self.gaze_indices + self.au_indices:
                    au_gaze_features[:, :, au_gaze_idx] = x[:, :, idx]
                    au_gaze_idx += 1
            else:
                # Without pose: split AU/gaze features between pathways
                all_features = self.gaze_indices + self.au_indices
                split_point = len(all_features) // 3  # Give 1/3 to xLSTM pathway
                
                # First third goes to xLSTM pathway (if it needs temporal dynamics)
                xlstm_indices = all_features[:split_point]
                # Remaining goes to CNN pathway
                cnn_indices = all_features[split_point:]
                
                pose_features = torch.zeros(batch_size, seq_len, len(xlstm_indices), device=x.device)
                au_gaze_features = torch.zeros(batch_size, seq_len, len(cnn_indices), device=x.device)
                
                # Fill features for xLSTM pathway
                for i, idx in enumerate(xlstm_indices):
                    pose_features[:, :, i] = x[:, :, idx]
                
                # Fill features for CNN pathway
                for i, idx in enumerate(cnn_indices):
                    au_gaze_features[:, :, i] = x[:, :, idx]
            
            return pose_features, au_gaze_features, audio_features
        else:
            # When not using hybrid pathways, collect all available features
            all_indices = []
            if self.include_pose:
                all_indices.extend(self.pose_indices)
            all_indices.extend(self.gaze_indices + self.au_indices)
            
            au_gaze_features = torch.zeros(batch_size, seq_len, len(all_indices), device=x.device)
            
            for i, idx in enumerate(all_indices):
                au_gaze_features[:, :, i] = x[:, :, idx]
            
            return None, au_gaze_features, audio_features
    
    def forward(self, x, audio_features=None, return_intermediates=False):
        """
        Forward pass with separate pathways:
        - xLSTM for pose features 
        - CNN for AU/gaze features
        - Audio xLSTM for audio features
        
        Args:
            x (torch.Tensor): Visual input features [batch, seq_length, input_size]
            audio_features (torch.Tensor, optional): Audio features if provided separately
            return_intermediates (bool): Whether to return intermediate activations
            
        Returns:
            torch.Tensor or tuple: Binary classification logits, or tuple with intermediates
        """
        # x shape: [batch, seq_length, input_size]
        batch_size, seq_len, _ = x.shape
        
        # Get audio features if passed as separate input
        if audio_features is None and self.include_audio and hasattr(x, 'audio_features'):
            # This handles the case where audio features are in the input dictionary
            audio_features = x.audio_features
        
        # Split features into groups
        pose_features, au_gaze_features, audio_features = self._split_features(x, audio_features)
        
        intermediates = {
            'input': x, 
            'au_gaze_features': au_gaze_features,
            'audio_features': audio_features
        }
        
        if self.use_hybrid_pathways:
            # Use multiple pathways
            
            # Apply targeted dropout during training
            if self.include_pose and self.training and self.pose_dropout > 0:
                pose_features = self.pose_feature_dropout(pose_features, self.training)
                intermediates['pose_features_after_dropout'] = pose_features
                
            # Process features with respective encoders
            pose_encoded = self.pose_encoder(pose_features)
            au_gaze_encoded = self.au_encoder(au_gaze_features)
            
            # Process audio if included
            audio_encoded = None
            if self.include_audio and audio_features is not None and hasattr(self, 'audio_encoder'):
                if self.training and self.audio_dropout > 0:
                    audio_features = self.audio_feature_dropout(audio_features, self.training)
                audio_encoded = self.audio_encoder(audio_features)
                intermediates['audio_encoded'] = audio_encoded
            
            # FUSION MECHANISM BASED ON TYPE
            if self.fusion_type == "cross_modal":
                # Cross-modal transformer fusion
                # Each modality attends to all other modalities
                pose_enhanced = pose_encoded
                au_gaze_enhanced = au_gaze_encoded
                
                # Enhance pose with information from AU/gaze
                pose_enhanced = self.cm_pose_au(pose_encoded, au_gaze_encoded)
                
                # Enhance AU/gaze with information from pose
                au_gaze_enhanced = self.cm_au_pose(au_gaze_encoded, pose_encoded)
                
                # If audio is included, perform cross-modal attention with it too
                if audio_encoded is not None:
                    # Enhance pose with audio information
                    pose_enhanced = self.cm_pose_audio(pose_enhanced, audio_encoded)
                    
                    # Enhance AU/gaze with audio information
                    au_gaze_enhanced = self.cm_au_audio(au_gaze_enhanced, audio_encoded)
                    
                    # Enhance audio with pose and AU/gaze information
                    audio_enhanced = self.cm_audio_pose(audio_encoded, pose_encoded)
                    audio_enhanced = self.cm_audio_au(audio_enhanced, au_gaze_encoded)
                    
                    # Merge all three enhanced encodings
                    merged = torch.cat([pose_enhanced, au_gaze_enhanced, audio_enhanced], dim=2)
                else:
                    # Merge the two enhanced encodings
                    merged = torch.cat([pose_enhanced, au_gaze_enhanced], dim=2)
                
                # Apply merger
                merged = self.merger_dropout(self.feature_merger(merged))
                intermediates['merged'] = merged
            else:
                # Original attention-based fusion
                # BALANCED FUSION: Get representations from all pathways
                
                # Apply attention to AU/gaze features
                au_gaze_attended = self.au_encoder.apply_attention(au_gaze_encoded)
                
                # Use last timestep of pose encoding
                pose_attended = pose_encoded[:, -1, :]
                
                # Project representations to a common dimension
                pose_proj = self.pose_projection(pose_attended)
                au_gaze_proj = self.au_projection(au_gaze_attended)
                
                if self.include_audio and audio_encoded is not None and hasattr(self, 'audio_projection'):
                    # Apply attention to audio features
                    audio_attended = self.audio_encoder.apply_attention(audio_encoded)
                    audio_proj = self.audio_projection(audio_attended)
                    
                    # Three-way fusion
                    combined_proj = torch.cat([pose_proj, au_gaze_proj, audio_proj], dim=1)
                    attention_weights = F.softmax(self.feature_attention(combined_proj), dim=-1)
                    self.last_modality_weights = attention_weights.detach()
                    
                    # Merge all three encodings
                    merged = torch.cat([pose_encoded, au_gaze_encoded, audio_encoded], dim=2)
                else:
                    # Two-way fusion
                    combined_proj = torch.cat([pose_proj, au_gaze_proj], dim=1)
                    attention_weights = F.softmax(self.feature_attention(combined_proj), dim=-1)
                    self.last_modality_weights = attention_weights.detach()
                    
                    # Merge the two encodings
                    merged = torch.cat([pose_encoded, au_gaze_encoded], dim=2)
                
                intermediates['modality_attention'] = attention_weights
                
                # Apply merger
                merged = self.merger_dropout(self.feature_merger(merged))
                intermediates['merged'] = merged
            
            # Apply global attention pooling
            attn_scores = self.global_attention(merged).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)
            self.last_attention_weights = attn_weights.detach()
            
            # Weighted sum based on attention
            pooled = torch.sum(merged * attn_weights, dim=1)
            
        else:
            # If not using hybrid pathways, process visual features through a single encoder
            cnn_output = self.au_gaze_encoder(au_gaze_features)
            
            if self.include_audio and audio_features is not None and hasattr(self, 'audio_encoder'):
                # Process audio separately
                audio_output = self.audio_encoder(audio_features)
                
                # Apply global attention to each pathway
                visual_attended = torch.sum(cnn_output * F.softmax(
                    self.global_attention(cnn_output).squeeze(-1), dim=1).unsqueeze(2), dim=1)
                
                audio_attended = self.audio_encoder.apply_attention(audio_output)
                
                # Project to common dimension
                visual_proj = self.visual_projection(visual_attended)
                audio_proj = self.audio_projection(audio_attended)
                
                # Two-way fusion for visual and audio
                combined_proj = torch.cat([visual_proj, audio_proj], dim=1)
                attention_weights = F.softmax(self.feature_attention(combined_proj), dim=-1)
                self.last_modality_weights = attention_weights.detach()
                
                # Combine visual and audio pathways
                merged = torch.cat([cnn_output, audio_output], dim=2)
                merged = self.merger_dropout(self.feature_merger(merged))
                
                # Apply global attention pooling
                attn_scores = self.global_attention(merged).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)
                self.last_attention_weights = attn_weights.detach()
                
                # Weighted sum based on attention
                pooled = torch.sum(merged * attn_weights, dim=1)
                
                intermediates['cnn_output'] = cnn_output
                intermediates['audio_output'] = audio_output
                intermediates['modality_attention'] = attention_weights
            else:
                # Visual-only pathway
                # Apply global attention pooling
                attn_scores = self.global_attention(cnn_output).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)
                self.last_attention_weights = attn_weights.detach()
                
                # Weighted sum based on attention
                pooled = torch.sum(cnn_output * attn_weights, dim=1)
                
                intermediates['cnn_output'] = cnn_output
            
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
            
        # Get xLSTM attention weights if available
        if hasattr(self, 'pose_encoder') and hasattr(self.pose_encoder, 'xlstm_stack'):
            for block_idx, block in enumerate(self.pose_encoder.xlstm_stack.blocks):
                # Check if the block has mlstm
                if hasattr(block, 'mlstm') and hasattr(block.mlstm, 'attn'):
                    if hasattr(block.mlstm.attn, 'last_attn_weights'):
                        attention_weights.append({
                            'layer': f'xlstm_block_{block_idx}_mlstm',
                            'weights': block.mlstm.attn.last_attn_weights.detach()
                        })
                
                # Check if the block has slstm
                if hasattr(block, 'slstm') and hasattr(block.slstm, 'attn'):
                    if hasattr(block.slstm.attn, 'last_attn_weights'):
                        attention_weights.append({
                            'layer': f'xlstm_block_{block_idx}_slstm',
                            'weights': block.slstm.attn.last_attn_weights.detach()
                        })
        
        return attention_weights
    
    # Feature importance methods (same as the previous implementations)
    
    def gradient_feature_importance(self, x, audio_features=None, target_class=1):
        """
        Calculate feature importance using gradient-based attribution
        
        Args:
            x (torch.Tensor): Visual input features
            audio_features (torch.Tensor, optional): Audio features if available
            target_class (int): Target class for attribution (1=depressed, 0=non-depressed)
            
        Returns:
            dict: Dictionary mapping feature names to importance scores
        """
        # Set model to eval mode
        self.eval()
        x_input = x.clone().detach().requires_grad_(True)
        
        # Clone audio features if provided and enable gradients
        audio_input = None
        if audio_features is not None:
            audio_input = audio_features.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self(x_input, audio_input)
        
        # Select target based on class - use logits directly
        if target_class == 1:
            # For positive class: maximize logits
            target = logits
        else:
            # For negative class: minimize logits
            target = -logits
        
        # Compute gradients w.r.t inputs
        if audio_input is not None:
            gradients, audio_gradients = torch.autograd.grad(
                outputs=target.sum(),
                inputs=[x_input, audio_input],
                create_graph=False,
                retain_graph=False
            )
        else:
            gradients = torch.autograd.grad(
                outputs=target.sum(),
                inputs=x_input,
                create_graph=False,
                retain_graph=False
            )[0]
        
        # Feature importance is the mean absolute gradient per feature
        feature_importance = torch.mean(torch.abs(gradients), dim=(0, 1)).detach().cpu().numpy()
        
        # Normalize importance scores
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        # Store feature importance
        if 'global' not in self.feature_importances:
            self.feature_importances['global'] = {}
            
        self.feature_importances['global']['gradient_based'] = feature_importance
        
        # Process audio importance if available
        audio_importance = None
        if audio_input is not None:
            audio_importance = torch.mean(torch.abs(audio_gradients), dim=(0, 1)).detach().cpu().numpy()
            if np.sum(audio_importance) > 0:
                audio_importance = audio_importance / np.sum(audio_importance)
            self.feature_importances['global']['gradient_based_audio'] = audio_importance
        
        # Map feature importance to feature names
        importance_dict = {
            self.feature_names[i]: feature_importance[i] 
            for i in range(min(len(self.feature_names), len(feature_importance)))
        }
        
        # Add audio feature importance if available
        if audio_importance is not None and self.audio_feature_names:
            for i, name in enumerate(self.audio_feature_names):
                if i < len(audio_importance):
                    importance_dict[name] = audio_importance[i]
        
        return importance_dict
    
    def integrated_gradients(self, x, audio_features=None, baseline=None, steps=50, target_class=1):
        """Calculate feature importance using integrated gradients"""
        # Set model to eval mode and ensure we're working with PyTorch tensors
        self.eval()
        x = x.detach()
        
        # Create baseline if not provided (zeros)
        if baseline is None:
            baseline = torch.zeros_like(x)
            if audio_features is not None:
                audio_baseline = torch.zeros_like(audio_features)
            else:
                audio_baseline = None
        else:
            audio_baseline = torch.zeros_like(audio_features) if audio_features is not None else None
        
        # Scale inputs along path from baseline to input
        scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)]
        if audio_features is not None:
            scaled_audio = [audio_baseline + (float(i) / steps) * (audio_features - audio_baseline) 
                           for i in range(steps + 1)]
        else:
            scaled_audio = [None] * (steps + 1)
        
        # Calculate gradients at each step
        total_gradients = torch.zeros_like(x)
        if audio_features is not None:
            total_audio_gradients = torch.zeros_like(audio_features)
        
        for i in range(steps + 1):
            scaled_input = scaled_inputs[i].clone().detach().requires_grad_(True)
            scaled_audio_input = scaled_audio[i].clone().detach().requires_grad_(True) if scaled_audio[i] is not None else None
            
            # Forward pass
            logits = self(scaled_input, scaled_audio_input)
            
            # Determine target based on class
            if target_class == 1:
                target = logits  # We want positive logits for "depressed" class
            else:
                target = -logits  # We want negative logits for "not depressed" class
            
            # Compute gradients for this step
            if scaled_audio_input is not None:
                grad, audio_grad = torch.autograd.grad(
                    outputs=target.sum(),
                    inputs=[scaled_input, scaled_audio_input],
                    create_graph=False,
                    retain_graph=False
                )
                total_audio_gradients += audio_grad.detach()
            else:
                grad = torch.autograd.grad(
                    outputs=target.sum(),
                    inputs=scaled_input,
                    create_graph=False,
                    retain_graph=False
                )[0]
            
            # Accumulate gradients
            total_gradients += grad.detach()
        
        # Average the gradients
        avg_gradients = total_gradients / (steps + 1)
        
        # Element-wise multiply average gradients with the input-baseline difference
        attributions = (x - baseline) * avg_gradients
        
        # Do the same for audio if present
        if audio_features is not None:
            avg_audio_gradients = total_audio_gradients / (steps + 1)
            audio_attributions = (audio_features - audio_baseline) * avg_audio_gradients
        
        # Sum over batch and sequence dimensions
        visual_importance = torch.mean(torch.abs(attributions), dim=(0, 1)).cpu().numpy()
        
        # Normalize if possible
        if np.sum(visual_importance) > 0:
            visual_importance = visual_importance / np.sum(visual_importance)
        
        # Process audio importance if available
        if audio_features is not None:
            audio_importance = torch.mean(torch.abs(audio_attributions), dim=(0, 1)).cpu().numpy()
            if np.sum(audio_importance) > 0:
                audio_importance = audio_importance / np.sum(audio_importance)
        else:
            audio_importance = None
        
        # Store in feature importances
        if 'global' not in self.feature_importances:
            self.feature_importances['global'] = {}
        
        self.feature_importances['global']['integrated_gradients_visual'] = visual_importance
        if audio_importance is not None:
            self.feature_importances['global']['integrated_gradients_audio'] = audio_importance
        
        # Map to feature names
        importance_dict = {
            self.feature_names[i]: visual_importance[i] 
            for i in range(min(len(self.feature_names), len(visual_importance)))
        }
        
        # Add audio feature importance if available
        if audio_importance is not None and self.audio_feature_names:
            audio_importance_dict = {
                self.audio_feature_names[i]: audio_importance[i]
                for i in range(min(len(self.audio_feature_names), len(audio_importance)))
            }
            # Combine visual and audio importance
            importance_dict.update(audio_importance_dict)
        
        return importance_dict
    
    def instance_feature_importance(self, x, audio_features=None):
        """Calculate instance-specific feature importance
        
        Args:
            x (torch.Tensor): Visual input features
            audio_features (torch.Tensor, optional): Audio features if available
            
        Returns:
            dict: Dictionary containing prediction and feature importance
        """
        # Get prediction
        with torch.no_grad():
            logits = self(x, audio_features)
            prob = torch.sigmoid(logits)
            pred_class = 1 if prob.item() > 0.5 else 0
        
        # Calculate feature importance for predicted class
        importance = self.integrated_gradients(x, audio_features=audio_features, target_class=pred_class)
        
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
                                   title="Top Feature Importance for Depression Detection",
                                   highlight_audio=True):
        """
        Visualize feature importance including audio features if available
        
        Args:
            importance_dict: Dictionary of feature importances
            top_k: Number of top features to show
            save_path: Path to save visualization
            title: Plot title
            highlight_audio: Whether to highlight audio features
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
            
            importance_dict = {
                self.feature_names[i]: importance_array[i]
                for i in range(min(len(self.feature_names), len(importance_array)))
            }
            
            # Add audio feature importance if available
            if hasattr(self, 'audio_feature_names') and len(self.audio_feature_names) > 0:
                if 'integrated_gradients_audio' in self.feature_importances['global']:
                    audio_importance = self.feature_importances['global']['integrated_gradients_audio']
                    for i, name in enumerate(self.audio_feature_names):
                        if i < len(audio_importance):
                            importance_dict[name] = audio_importance[i]
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k features
        top_features = sorted_features[:top_k]
        feature_names = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(feature_names)), importance_values, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Importance')
        plt.title(title)
        
        # Highlight features by type
        au_color = 'red'
        pose_color = 'green'
        audio_color = 'purple'
        default_color = 'cornflowerblue'
        
        has_audio = False
        has_aus = False
        has_pose = False
        
        # Clinically significant AUs
        highlighted_aus = ['AU01', 'AU04', 'AU05', 'AU06', 'AU07', 'AU12', 'AU15']
        
        for i, feature in enumerate(feature_names):
            # Check if it's an audio feature
            if highlight_audio and feature in self.audio_feature_names:
                bars[i].set_color(audio_color)
                has_audio = True
            # Check if it's a clinically significant AU
            elif any(au in feature for au in highlighted_aus):
                bars[i].set_color(au_color)
                has_aus = True
            # Check if it's a pose feature
            elif 'pose_' in feature:
                bars[i].set_color(pose_color)
                has_pose = True
            # Default color for other features
            else:
                bars[i].set_color(default_color)
        
        # Add legend to explain colors
        from matplotlib.patches import Patch
        legend_elements = []
        
        if has_aus:
            legend_elements.append(Patch(facecolor=au_color, label='Clinically Significant AUs'))
        if has_pose:
            legend_elements.append(Patch(facecolor=pose_color, label='Pose Features'))
        if has_audio:
            legend_elements.append(Patch(facecolor=audio_color, label='Audio Features'))
        
        legend_elements.append(Patch(facecolor=default_color, label='Other Features'))
        
        # Only show legend if we have at least one highlighted feature
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_modality_importance(self):
        """Get relative importance of each modality with context-aware labels"""
        if not hasattr(self, 'last_modality_weights'):
            return None
        
        weights = self.last_modality_weights.cpu().numpy().mean(axis=0)
        
        # Create appropriate pathway labels based on what features are actually being processed
        if self.include_audio and hasattr(self, 'audio_encoder') and len(weights) == 3:
            # Three-pathway model
            if self.include_pose and len(self.pose_indices) > 0:
                # When pose is included, first pathway processes actual pose data
                pathway_labels = {
                    'pose': weights[0],
                    'facial_expressions': weights[1],
                    'audio': weights[2]
                }
            else:
                # When pose is excluded, first pathway processes AU/gaze through xLSTM
                pathway_labels = {
                    'au_gaze_temporal': weights[0],  # AU/gaze through xLSTM pathway
                    'au_gaze_spatial': weights[1],   # AU/gaze through CNN pathway
                    'audio': weights[2]
                }
        else:
            # Two-pathway model
            if self.include_pose and len(self.pose_indices) > 0:
                # When pose is included
                pathway_labels = {
                    'pose': weights[0],
                    'facial_expressions': weights[1]
                }
            else:
                # When pose is excluded
                pathway_labels = {
                    'au_gaze_temporal': weights[0],  # AU/gaze through xLSTM pathway
                    'au_gaze_spatial': weights[1]    # AU/gaze through CNN pathway
                }
        
        return pathway_labels
    
    def get_clinical_au_importance(self):
        """Extract importance of clinically significant Action Units"""
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
    
    def get_audio_feature_importance(self):
        """Extract importance of audio features"""
        if not hasattr(self, 'audio_feature_names') or not self.audio_feature_names:
            return {}
            
        if self.feature_importances is None or 'global' not in self.feature_importances:
            raise ValueError("No feature importance data available")
            
        # Get audio-specific importance if available
        if 'integrated_gradients_audio' in self.feature_importances['global']:
            audio_importance = self.feature_importances['global']['integrated_gradients_audio']
            return {
                self.audio_feature_names[i]: audio_importance[i]
                for i in range(min(len(self.audio_feature_names), len(audio_importance)))
            }
        else:
            # Try to extract from global importance
            global_importance = self.feature_importances['global'].get('integrated_gradients', 
                                   next(iter(self.feature_importances['global'].values())))
            
            # Return empty dict if dimensions don't match
            if len(global_importance) < len(self.feature_names) + len(self.audio_feature_names):
                return {}
                
            # Extract audio feature importance from the end of the array
            audio_importance = global_importance[len(self.feature_names):]
            return {
                self.audio_feature_names[i]: audio_importance[i]
                for i in range(min(len(self.audio_feature_names), len(audio_importance)))
            }
