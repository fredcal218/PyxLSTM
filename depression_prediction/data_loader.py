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
from sklearn.preprocessing import StandardScaler


class DepressionDataset(Dataset):
    """
    Dataset for depression severity prediction using pose, gaze, and AU features.
    
    Args:
        data_dir (str): Directory containing the E-DAIC data.
        split (str): One of 'train', 'dev', or 'test'.
        max_seq_length (int, optional): Maximum sequence length. Default: 1000.
        stride (int, optional): Stride for sequence sampling. Default: 500.
    """

    def __init__(self, data_dir, split, max_seq_length=1000, stride=500):
        self.data_dir = data_dir
        self.split = split
        self.max_seq_length = max_seq_length
        self.stride = stride
        
        self.labels_path = os.path.join(data_dir, "labels", f"{split}_split.csv")
        self.features_dir = os.path.join(data_dir, "data_extr", split)
        
        # Load labels
        self.labels_df = pd.read_csv(self.labels_path)
        
        # Process participant IDs and create sequences
        self.sequences = self._create_sequences()
        
        # Initialize feature scaler (will be fit during first data loading)
        self.scaler = None

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

    def _load_and_preprocess_features(self, features_path):
        """Load and preprocess features from CSV file."""
        # Load features
        df = pd.read_csv(features_path)
        
        # Select pose, gaze, and AU columns (excluding timestamp, confidence, etc.)
        # Explicitly exclude "_c" columns which contain confidence values for AUs
        feature_cols = [col for col in df.columns if 
                        (any(prefix in col for prefix in ['pose', 'gaze', 'AU']) and 
                         not col.endswith('_c'))]
        features = df[feature_cols].fillna(0).values
        
        # Initialize and fit scaler if needed
        if self.scaler is None and self.split == 'train':
            self.scaler = StandardScaler()
            self.scaler.fit(features)
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
            
        return features

    def __len__(self):
        """Return the number of sequences."""
        total_sequences = 0
        for entry in self.sequences:
            # Load features for this calculation
            features = self._load_and_preprocess_features(entry['features_path'])
            # Calculate number of possible sequences with stride
            possible_sequences = max(1, (len(features) - self.max_seq_length) // self.stride + 1)
            total_sequences += possible_sequences
        return total_sequences

    def __getitem__(self, idx):
        """Get a sequence by index."""
        # Find the corresponding participant and sequence index
        current_count = 0
        for entry in self.sequences:
            features = self._load_and_preprocess_features(entry['features_path'])
            possible_sequences = max(1, (len(features) - self.max_seq_length) // self.stride + 1)
            
            if idx < current_count + possible_sequences:
                # This is the participant we want
                sequence_idx = idx - current_count
                start_idx = sequence_idx * self.stride
                end_idx = min(start_idx + self.max_seq_length, len(features))
                
                # Extract sequence
                sequence = features[start_idx:end_idx]
                
                # Pad if necessary
                if len(sequence) < self.max_seq_length:
                    padding = np.zeros((self.max_seq_length - len(sequence), sequence.shape[1]))
                    sequence = np.vstack([sequence, padding])
                
                # Convert to tensor
                sequence_tensor = torch.FloatTensor(sequence)
                label_tensor = torch.FloatTensor([entry['phq8_score']])
                
                return sequence_tensor, label_tensor
            
            current_count += possible_sequences
        
        raise IndexError("Index out of bounds")


def get_data_loaders(data_dir, batch_size=16, max_seq_length=1000, stride=500):
    """
    Create data loaders for train, dev, and test sets.
    
    Args:
        data_dir (str): Directory containing the E-DAIC data.
        batch_size (int, optional): Batch size. Default: 16.
        max_seq_length (int, optional): Maximum sequence length. Default: 1000.
        stride (int, optional): Stride for sequence sampling. Default: 500.
    
    Returns:
        tuple: Train, dev, and test data loaders.
    """
    print("Creating training dataset...")
    start_time = time.time()
    train_dataset = DepressionDataset(data_dir, 'train', max_seq_length, stride)
    print(f"Training dataset created in {time.time() - start_time:.2f} seconds")

    print("Creating dev dataset...")
    start_time = time.time()
    dev_dataset = DepressionDataset(data_dir, 'dev', max_seq_length, stride)
    print(f"Dev dataset created in {time.time() - start_time:.2f} seconds")

    print("Creating test dataset...")
    start_time = time.time()
    test_dataset = DepressionDataset(data_dir, 'test', max_seq_length, stride)
    print(f"Test dataset created in {time.time() - start_time:.2f} seconds")

    # Share the scaler from train dataset with dev and test datasets
    dev_dataset.scaler = train_dataset.scaler
    test_dataset.scaler = train_dataset.scaler
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader, test_loader
