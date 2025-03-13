"""
Data loader for depression severity prediction using pose, gaze, and AU features.

This module provides functionality to load and preprocess the E-DAIC dataset
for depression severity prediction using the PHQ-8 scale.
"""

import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler  # Changed from StandardScaler
import pickle
from depression_prediction.sequence_utils import create_overlapping_windows


def custom_collate_fn(batch):
    """
    Custom collate function to handle sequences of varying lengths.
    
    This collate function pads sequences to the max length in the batch,
    which is more efficient than padding all sequences to the global max length.
    
    Args:
        batch (list): List of tuples (features, label)
        
    Returns:
        tuple: Batched features and labels
    """
    # Find max sequence length in this batch
    max_len = max([item[0].size(0) for item in batch])
    
    # Get feature dimension
    feature_dim = batch[0][0].size(1)
    
    # Pad sequences to max length in batch
    features = []
    labels = []
    
    for feature, label in batch:
        # Current sequence length
        seq_len = feature.size(0)
        
        if seq_len < max_len:
            # Create padding
            padding = torch.zeros(max_len - seq_len, feature_dim, dtype=feature.dtype)
            # Concatenate padding
            padded_feature = torch.cat([feature, padding], dim=0)
            features.append(padded_feature)
        else:
            features.append(feature)
            
        labels.append(label)
    
    # Stack into batch tensors
    features = torch.stack(features)
    labels = torch.stack(labels)
    
    return features, labels


class DepressionDataset(Dataset):
    """
    Dataset for depression severity prediction using pose, gaze, and AU features.
    
    Args:
        data_dir (str): Directory containing the E-DAIC data.
        split (str): One of 'train', 'dev', or 'test'.
        max_seq_length (int, optional): Maximum sequence length. Default: 1000.
        stride (int, optional): Stride for sequence sampling. Default: 500.
        cache_features (bool, optional): Whether to cache features. Default: True.
    """

    def __init__(self, data_dir, split, max_seq_length=1000, stride=500, cache_features=True):
        self.data_dir = data_dir
        self.split = split
        self.max_seq_length = max_seq_length
        self.stride = stride
        self.cache_features = cache_features
        
        self.labels_path = os.path.join(data_dir, "labels", f"{split}_split.csv")
        self.features_dir = os.path.join(data_dir, "data_extr", split)
        self.cache_dir = os.path.join(data_dir, "cache")
        
        # Create cache directory if needed
        if self.cache_features and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Load labels
        self.labels_df = pd.read_csv(self.labels_path)
        
        # Process participant IDs and create sequences
        self.sequences = self._create_sequences()
        
        # Initialize feature scaler (will be fit during first data loading)
        self.scaler = None
        
        # Cache for features
        self.feature_cache = {}
        
        # Pre-calculate total sequence count
        self._precalculate_sequence_count()

    def _create_sequences(self):
        """Create sequences from participant data."""
        sequences = []
        
        for _, row in self.labels_df.iterrows():
            participant_id = row['Participant_ID']
            phq8_score = row['PHQ_Score']
            
            # Path to participant's features
            participant_dir = os.path.join(self.features_dir, f"{participant_id}_P")
            features_path = os.path.join(participant_dir, "features", 
                                         f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")
            
            if not os.path.exists(features_path):
                print(f"Warning: Features not found for participant {participant_id}")
                continue
            
            # Add entry to sequences list with participant info
            sequences.append({
                'participant_id': participant_id,
                'features_path': features_path,
                'phq8_score': phq8_score
            })
        
        return sequences
    
    def _get_cache_path(self, participant_id):
        """Get path for cached features."""
        return os.path.join(self.cache_dir, f"{self.split}_{participant_id}_features.pkl")

    def _load_and_preprocess_features(self, features_path, participant_id=None):
        """Load and preprocess features from CSV file with caching."""
        # Check cache first if participant_id is provided and caching is enabled
        if self.cache_features and participant_id:
            cache_path = self._get_cache_path(participant_id)
            
            # Return cached features if available
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Load features
        df = pd.read_csv(features_path)
        
        # Select pose, gaze, and AU columns (excluding timestamp, confidence, etc.)
        feature_cols = [col for col in df.columns if 
                        (any(prefix in col for prefix in ['pose', 'gaze', 'AU']) and 
                         not col.endswith('_c'))]
        
        # Use only valid columns (ensures we don't include invalid columns)
        valid_cols = [col for col in feature_cols if col in df.columns]
        features = df[valid_cols].values
        
        # Time-based frequency analysis for noise reduction
        # Use rolling windows to smooth out high-frequency noise
        if features.shape[0] > 10:  # Only if we have enough frames
            df_features = pd.DataFrame(features)
            # Apply rolling median with a small window
            df_features = df_features.rolling(window=5, min_periods=1, center=True).median()
            features = df_features.values
        
        # Handle NaNs with forward-fill then backward-fill
        if features.shape[0] > 0:
            df_features = pd.DataFrame(features)
            df_features = df_features.fillna(method='ffill').fillna(method='bfill')
            # Any remaining NaNs, fill with column median
            for col_idx in range(df_features.shape[1]):
                col_data = df_features.iloc[:, col_idx]
                if col_data.isna().any():
                    median_val = col_data.median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_features.iloc[:, col_idx] = col_data.fillna(median_val)
            
            features = df_features.values
        
        # Handle outliers with quantile-based clipping
        percentile_95 = np.percentile(features, 95, axis=0)
        percentile_05 = np.percentile(features, 5, axis=0)
        features = np.clip(features, percentile_05, percentile_95)
        
        # Feature normalization
        if self.scaler is None and self.split == 'train':
            self.scaler = RobustScaler(quantile_range=(25.0, 75.0))  # Standard IQR
            self.scaler.fit(features)
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Final safety check
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Cache the processed features
        if self.cache_features and participant_id:
            cache_path = self._get_cache_path(participant_id)
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        
        return features

    def _precalculate_sequence_count(self):
        """Pre-calculate the total number of sequences to avoid reloading features in __len__."""
        self.total_sequences = 0
        self.sequence_indices = []
        
        for i, entry in enumerate(self.sequences):
            participant_id = entry['participant_id']
            
            # Load features once for this calculation
            features = self._load_and_preprocess_features(
                entry['features_path'], participant_id)
            
            # Calculate number of possible sequences with stride
            possible_sequences = max(1, (len(features) - self.max_seq_length) // self.stride + 1)
            
            # Store mapping for faster indexing in __getitem__
            self.sequence_indices.extend([(i, j) for j in range(possible_sequences)])
            self.total_sequences += possible_sequences

    def __len__(self):
        """Return the number of sequences."""
        return self.total_sequences

    def __getitem__(self, idx):
        """Get a sequence by index using precalculated indices."""
        if idx >= len(self.sequence_indices):
            raise IndexError("Index out of bounds")
            
        # Get the participant index and sequence index
        participant_idx, sequence_idx = self.sequence_indices[idx]
        entry = self.sequences[participant_idx]
        
        # Load features for this participant
        participant_id = entry['participant_id']
        features = self._load_and_preprocess_features(entry['features_path'], participant_id)
        
        # Calculate start and end indices for the sequence
        start_idx = sequence_idx * self.stride
        end_idx = min(start_idx + self.max_seq_length, len(features))
        
        # Extract sequence
        sequence = features[start_idx:end_idx]
        
        # For long sequences, use dynamic downsampling to reduce memory usage
        if len(sequence) > self.max_seq_length:
            # Downsample by taking every nth frame
            n = len(sequence) // self.max_seq_length + 1
            sequence = sequence[::n][:self.max_seq_length]
        
        # Always ensure sequence length is exactly max_seq_length
        if len(sequence) < self.max_seq_length:
            # Use reflection padding for better continuity if we have enough data
            if len(sequence) > 10:
                pad_size = self.max_seq_length - len(sequence)
                reflection_padding = []
                
                for i in range(min(pad_size, len(sequence))):
                    reflection_padding.append(sequence[len(sequence)-1-i % len(sequence)])
                    
                padding = np.array(reflection_padding[:pad_size])
                sequence = np.vstack([sequence, padding])
            else:
                # Use zero padding for very short sequences
                padding = np.zeros((self.max_seq_length - len(sequence), sequence.shape[1]))
                sequence = np.vstack([sequence, padding])
        
        # Convert to tensor and ensure it's a valid float32
        sequence_tensor = torch.FloatTensor(sequence).contiguous()
        label_tensor = torch.FloatTensor([entry['phq8_score']])
        
        # Final check for NaNs or infinities
        if torch.isnan(sequence_tensor).any() or torch.isinf(sequence_tensor).any():
            sequence_tensor = torch.nan_to_num(sequence_tensor)
        
        return sequence_tensor, label_tensor


def get_data_loaders(data_dir, batch_size=16, max_seq_length=1000, stride=500, num_workers=4):
    """
    Create data loaders for train, dev, and test sets.
    
    Args:
        data_dir (str): Directory containing the E-DAIC data.
        batch_size (int, optional): Batch size. Default: 16.
        max_seq_length (int, optional): Maximum sequence length. Default: 1000.
        stride (int, optional): Stride for sequence sampling. Default: 500.
        num_workers (int, optional): Number of worker processes. Default: 4.
    
    Returns:
        tuple: Train, dev, and test data loaders.
    """
    print("Creating training dataset...")
    start_time = time.time()
    train_dataset = DepressionDataset(data_dir, 'train', max_seq_length, stride, cache_features=True)
    print(f"Training dataset created in {time.time() - start_time:.2f} seconds")

    print("Creating dev dataset...")
    start_time = time.time()
    dev_dataset = DepressionDataset(data_dir, 'dev', max_seq_length, stride, cache_features=True)
    print(f"Dev dataset created in {time.time() - start_time:.2f} seconds")

    print("Creating test dataset...")
    start_time = time.time()
    test_dataset = DepressionDataset(data_dir, 'test', max_seq_length, stride, cache_features=True)
    print(f"Test dataset created in {time.time() - start_time:.2f} seconds")

    # Share the scaler from train dataset with dev and test datasets
    dev_dataset.scaler = train_dataset.scaler
    test_dataset.scaler = train_dataset.scaler
    
    # Use multiple workers for faster data loading - with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        collate_fn=custom_collate_fn  # Add custom collate function
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Add custom collate function
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Add custom collate function
    )
    
    return train_loader, dev_loader, test_loader
