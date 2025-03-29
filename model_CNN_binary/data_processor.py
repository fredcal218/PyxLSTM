import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from movement_analysis import MovementFeatureExtractor

class EDAICDataset(Dataset):
    """
    Dataset class for E-DAIC depression binary classification
    """
    def __init__(self, data_dir, labels_path, split='train', max_seq_length=150, confidence_threshold=0.9, 
                feature_scalers=None, is_train=False, include_movement_features=True, 
                include_pose=True, include_gaze=True, include_au=True):
        """
        Initialize E-DAIC dataset
        
        Args:
            data_dir (str): Directory containing feature data
            labels_path (str): Path to labels CSV file
            split (str): Dataset split ('train', 'dev', or 'test')
            max_seq_length (int): Maximum sequence length to use
            confidence_threshold (float): Minimum confidence value to include a frame
            feature_scalers (dict): Dictionary of feature scalers for normalization
            is_train (bool): Whether this is the training set (for fitting scalers)
            include_movement_features (bool): Whether to include derived movement features
            include_pose (bool): Whether to include pose features
            include_gaze (bool): Whether to include gaze features
            include_au (bool): Whether to include action unit features
        """
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.split = split
        self.confidence_threshold = confidence_threshold
        self.feature_scalers = feature_scalers
        self.is_train = is_train
        self.include_pose = include_pose
        self.include_gaze = include_gaze
        self.include_au = include_au
        self.include_movement_features = include_movement_features and include_pose  # Movement features require pose
        
        # Initialize movement feature extractor only if needed
        if self.include_movement_features:
            self.movement_extractor = MovementFeatureExtractor()
        
        # Load labels
        self.labels_df = pd.read_csv(labels_path)
        
        # Get participant IDs for this split
        self.participant_ids = [str(pid) for pid in self.labels_df['Participant_ID'].tolist()]
        
        # Map participant IDs to binary depression labels
        self.binary_labels = {}
        self.phq_scores = {}  # Keep original scores for reference only
        
        for _, row in self.labels_df.iterrows():
            pid = str(row['Participant_ID'])
            self.binary_labels[pid] = row['PHQ_Binary']
            
            # Store PHQ score for reference if available
            if 'PHQ8_Score' in row:
                self.phq_scores[pid] = row['PHQ8_Score']
        
        # Collect feature files
        self.feature_files = []
        self.valid_participants = []
        
        for pid in self.participant_ids:
            feature_path = os.path.join(data_dir, f"{split}/{pid}_P/features/{pid}_OpenFace2.1.0_Pose_Gaze_AUs.csv")
            if os.path.exists(feature_path):
                self.feature_files.append(feature_path)
                self.valid_participants.append(pid)
        
        # Extract feature names (used for interpretability)
        if len(self.feature_files) > 0:
            sample_df = pd.read_csv(self.feature_files[0])
            
            # Get only the requested feature types
            self.feature_names = []
            
            for col in sample_df.columns:
                # Include pose features if requested
                if col.startswith('pose_'):
                    if self.include_pose:
                        self.feature_names.append(col)
                # Include gaze features if requested
                elif col.startswith('gaze_'):
                    if self.include_gaze:
                        self.feature_names.append(col)
                # Include AU features if requested (excluding confidence values)
                elif col.startswith('AU') and not col.endswith('_c'):
                    if self.include_au:
                        self.feature_names.append(col)
            
            # If including movement features (requires pose to be included), extract them
            if self.include_movement_features:
                sample_with_movement = self.movement_extractor.extract_movement_features(
                    sample_df[self.feature_names].head(100))  # Use just head for efficiency
                movement_features = [col for col in sample_with_movement.columns 
                                    if col not in self.feature_names]
                self.feature_names.extend(movement_features)
                
            # Initialize feature scalers if we're the training set and no scalers provided
            if is_train and feature_scalers is None:
                self.feature_scalers = {feature: StandardScaler() for feature in self.feature_names}
            else:
                self.feature_scalers = feature_scalers
        else:
            self.feature_names = []
            print(f"Warning: No feature files found for {split} split")
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_path = self.feature_files[idx]
        pid = self.valid_participants[idx]
        
        # Load features
        df = pd.read_csv(feature_path)
        
        # Apply confidence filtering
        high_confidence_mask = df['confidence'] >= self.confidence_threshold
        
        # Filter frames with high confidence only
        if sum(high_confidence_mask) > 0:
            df = df[high_confidence_mask].reset_index(drop=True)
        else:
            print(f"Warning: No frames with confidence >= {self.confidence_threshold} for {pid}. Using all frames.")
        
        # Extract basic features (pose, gaze, AUs)
        base_features = [col for col in self.feature_names if col in df.columns]
        features_df = df[base_features].copy()
        
        # Add movement features if requested
        if self.include_movement_features:
            features_df = self.movement_extractor.extract_movement_features(features_df)
            
            # Make sure to only use features that were defined in feature_names
            # This ensures consistency across all participants
            features_df = features_df[self.feature_names]
        
        # Normalize features if scalers are available
        if self.feature_scalers:
            normalized_features = features_df.copy()
            
            # Apply scaling for each feature separately
            for feature in self.feature_names:
                if feature in features_df.columns:
                    feature_values = features_df[feature].values.reshape(-1, 1)
                    
                    # Fit scaler if this is training set, otherwise just transform
                    if self.is_train:
                        normalized_values = self.feature_scalers[feature].fit_transform(feature_values)
                    else:
                        normalized_values = self.feature_scalers[feature].transform(feature_values)
                        
                    normalized_features[feature] = normalized_values.flatten()
            
            features = normalized_features.values
        else:
            features = features_df.values
        
        # Handle sequence length
        if features.shape[0] > self.max_seq_length:
            # Truncate sequence
            features = features[:self.max_seq_length]
        elif features.shape[0] < self.max_seq_length:
            # Pad sequence
            padding = np.zeros((self.max_seq_length - features.shape[0], features.shape[1]))
            features = np.vstack((features, padding))
        
        # Get binary depression label
        binary_label = self.binary_labels[pid]
        
        return_dict = {
            'features': torch.FloatTensor(features),
            'binary_label': torch.FloatTensor([binary_label]),
            'participant_id': pid,
            'seq_length': min(features.shape[0], self.max_seq_length)
        }
        
        # Add PHQ score if available (for reference only)
        if pid in self.phq_scores:
            return_dict['phq_score'] = torch.FloatTensor([self.phq_scores[pid]])
            
        return return_dict
    
    def get_feature_names(self):
        """Get feature names for interpretability"""
        return self.feature_names
    
    def get_class_distribution(self):
        """Get distribution of depression classes"""
        depressed = sum(1 for pid in self.valid_participants 
                       if self.binary_labels[pid] == 1)
        non_depressed = len(self.valid_participants) - depressed
        
        return {
            'depressed': depressed,
            'non_depressed': non_depressed,
            'total': len(self.valid_participants)
        }
    
    def get_feature_scalers(self):
        """Get fitted feature scalers for transfer to other datasets"""
        return self.feature_scalers

def get_dataloaders(data_dir, labels_dir, batch_size=16, max_seq_length=150, 
                  confidence_threshold=0.9, include_movement_features=True, 
                  include_pose=True, include_gaze=True, include_au=True):
    """
    Create data loaders for train, dev, and test sets
    
    Args:
        data_dir (str): Base directory for E-DAIC data
        labels_dir (str): Directory containing label files
        batch_size (int): Batch size for dataloaders
        max_seq_length (int): Maximum sequence length
        confidence_threshold (float): Minimum confidence for face detection
        include_movement_features (bool): Whether to include derived movement features
        include_pose (bool): Whether to include pose features
        include_gaze (bool): Whether to include gaze features
        include_au (bool): Whether to include action unit features
        
    Returns:
        dict: Dictionary of dataloaders and datasets for train, dev, and test sets
    """
    # Create training dataset first to fit the normalizers
    train_dataset = EDAICDataset(
        data_dir=data_dir,
        labels_path=os.path.join(labels_dir, "train_split.csv"),
        split='train',
        max_seq_length=max_seq_length,
        confidence_threshold=confidence_threshold,
        feature_scalers=None,  # Will initialize new scalers
        is_train=True,  # Will fit the scalers
        include_movement_features=include_movement_features,
        include_pose=include_pose,
        include_gaze=include_gaze,
        include_au=include_au
    )
    
    # Pre-fit the scalers on all training data
    print("Fitting feature normalizers on training data...")
    all_feature_data = {}
    for feature in train_dataset.feature_names:
        all_feature_data[feature] = []
    
    # Collect feature values across all training participants
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        features = sample['features'].numpy()
        seq_length = sample['seq_length']
        
        # Only use valid sequence length (not padding)
        valid_features = features[:seq_length]
        
        # Add to collection for each feature
        for i, feature_name in enumerate(train_dataset.feature_names):
            all_feature_data[feature_name].extend(valid_features[:, i].tolist())
    
    # Fit each scaler on collected data
    feature_scalers = {}
    for feature in train_dataset.feature_names:
        if len(all_feature_data[feature]) > 0:
            scaler = StandardScaler()
            scaler.fit(np.array(all_feature_data[feature]).reshape(-1, 1))
            feature_scalers[feature] = scaler
    
    # Create datasets with fitted scalers
    train_dataset = EDAICDataset(
        data_dir=data_dir,
        labels_path=os.path.join(labels_dir, "train_split.csv"),
        split='train',
        max_seq_length=max_seq_length,
        confidence_threshold=confidence_threshold,
        feature_scalers=feature_scalers,
        is_train=False,  # Don't refit, just transform
        include_movement_features=include_movement_features,
        include_pose=include_pose,
        include_gaze=include_gaze,
        include_au=include_au
    )
    
    dev_dataset = EDAICDataset(
        data_dir=data_dir,
        labels_path=os.path.join(labels_dir, "dev_split.csv"),
        split='dev',
        max_seq_length=max_seq_length,
        confidence_threshold=confidence_threshold,
        feature_scalers=feature_scalers,
        is_train=False,
        include_movement_features=include_movement_features,
        include_pose=include_pose,
        include_gaze=include_gaze,
        include_au=include_au
    )
    
    test_dataset = EDAICDataset(
        data_dir=data_dir,
        labels_path=os.path.join(labels_dir, "test_split.csv"),
        split='test',
        max_seq_length=max_seq_length,
        confidence_threshold=confidence_threshold,
        feature_scalers=feature_scalers,
        is_train=False,
        include_movement_features=include_movement_features,
        include_pose=include_pose,
        include_gaze=include_gaze,
        include_au=include_au
    )
    
    # Print dataset statistics
    print("Dataset statistics:")
    for split, dataset in [('Train', train_dataset), ('Dev', dev_dataset), ('Test', test_dataset)]:
        dist = dataset.get_class_distribution()
        print(f"  {split}: {dist['total']} samples "
              f"({dist['depressed']} depressed, {dist['non_depressed']} non-depressed, "
              f"{dist['depressed']/dist['total']*100:.1f}% depressed)")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train': {'loader': train_loader, 'dataset': train_dataset},
        'dev': {'loader': dev_loader, 'dataset': dev_dataset},
        'test': {'loader': test_loader, 'dataset': test_dataset}
    }
