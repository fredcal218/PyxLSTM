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
                feature_scalers=None, is_train=False, include_movement_features=True, include_pose=True,
                include_audio=True):
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
            include_audio (bool): Whether to include audio (eGeMAPS) features
        """
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.split = split
        self.confidence_threshold = confidence_threshold
        self.feature_scalers = feature_scalers
        self.is_train = is_train
        self.include_movement_features = include_movement_features and include_pose  # Movement features require pose
        self.include_pose = include_pose
        self.include_audio = include_audio
        
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
        self.audio_files = []  # New list for audio feature files
        self.valid_participants = []
        
        for pid in self.participant_ids:
            visual_feature_path = os.path.join(data_dir, f"{split}/{pid}_P/features/{pid}_OpenFace2.1.0_Pose_Gaze_AUs.csv")
            audio_feature_path = os.path.join(data_dir, f"{split}/{pid}_P/features/{pid}_OpenSMILE2.3.0_egemaps.csv")
            
            # Check if both visual and audio features exist for the participant
            visual_exists = os.path.exists(visual_feature_path)
            audio_exists = os.path.exists(audio_feature_path) if self.include_audio else True
            
            if visual_exists and audio_exists:
                self.feature_files.append(visual_feature_path)
                if self.include_audio:
                    self.audio_files.append(audio_feature_path)
                self.valid_participants.append(pid)
            elif visual_exists and not self.include_audio:
                # If we're not requiring audio, include participants with only visual data
                self.feature_files.append(visual_feature_path)
                self.audio_files.append(None)  # Placeholder
                self.valid_participants.append(pid)
        
        # Extract feature names (used for interpretability)
        self.feature_names = []
        self.audio_feature_names = []
        
        if len(self.feature_files) > 0:
            sample_df = pd.read_csv(self.feature_files[0])
            
            # Get only the requested feature types
            for col in sample_df.columns:
                # Include pose features if requested
                if col.startswith('pose_'):
                    if self.include_pose:
                        self.feature_names.append(col)
                # Include gaze features
                elif col.startswith('gaze_'):
                    self.feature_names.append(col)
                # Include AU intensity features but explicitly exclude confidence columns
                elif col.startswith('AU') and col.endswith('_r'):  # Only include intensity (_r) columns
                    self.feature_names.append(col)
            
            # If including movement features (requires pose to be included), extract them
            if self.include_movement_features:
                sample_with_movement = self.movement_extractor.extract_movement_features(
                    sample_df[self.feature_names].head(100))  # Use just head for efficiency
                movement_features = [col for col in sample_with_movement.columns 
                                    if col not in self.feature_names]
                self.feature_names.extend(movement_features)
                
            # Get audio feature names if available and requested
            if self.include_audio and len(self.audio_files) > 0 and self.audio_files[0] is not None:
                try:
                    audio_sample_df = pd.read_csv(self.audio_files[0], sep=';')
                    # Exclude the name and frameTime columns and other non-feature columns
                    self.audio_feature_names = [col for col in audio_sample_df.columns 
                                              if col not in ['name', 'frameTime'] and
                                                 not col.startswith('frame') and 
                                                 not col.startswith('timestamp')]
                except Exception as e:
                    print(f"Error loading audio sample file: {e}")
                    self.audio_feature_names = []
            
            # Initialize feature scalers if we're the training set and no scalers provided
            if is_train and feature_scalers is None:
                # Create separate scalers for visual and audio features
                self.feature_scalers = {}
                # Add scalers for visual features
                for feature in self.feature_names:
                    self.feature_scalers[feature] = StandardScaler()
                # Add scalers for audio features
                for feature in self.audio_feature_names:
                    self.feature_scalers[feature] = StandardScaler()
            else:
                self.feature_scalers = feature_scalers
        else:
            print(f"Warning: No feature files found for {split} split")
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_path = self.feature_files[idx]
        audio_path = self.audio_files[idx] if self.include_audio and idx < len(self.audio_files) else None
        pid = self.valid_participants[idx]
        
        # Load visual features
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
            
            # Get only the features that exist in feature_names
            # This ensures consistency across all participants
            available_features = [col for col in self.feature_names if col in features_df.columns]
            features_df = features_df[available_features]
        
        # Load audio features if available
        audio_features = None
        if self.include_audio and audio_path is not None:
            try:
                # Make sure to use semicolon separator for audio files
                audio_df = pd.read_csv(audio_path, sep=';')
                
                # Extract only the feature columns (not name/frameTime)
                audio_feature_cols = [col for col in audio_df.columns if col in self.audio_feature_names]
                
                if audio_feature_cols:
                    audio_features_df = audio_df[audio_feature_cols]
                    
                    # Align audio features with visual features by interpolation
                    # Audio and visual might have different sampling rates
                    if len(audio_df) != len(df):
                        # Create a mapping from original audio frames to visual frames
                        if len(audio_df) > 1 and len(df) > 1:
                            # Create new indices for interpolation
                            orig_indices = np.linspace(0, 1, len(audio_df))
                            new_indices = np.linspace(0, 1, len(df))
                            
                            # Interpolate each audio feature to match visual frame count
                            audio_resampled = {}
                            for col in audio_feature_cols:
                                audio_resampled[col] = np.interp(
                                    new_indices, orig_indices, audio_df[col].values)
                            
                            audio_features_df = pd.DataFrame(audio_resampled)
                        else:
                            # If too few frames, duplicate the single frame
                            audio_features_df = pd.DataFrame({
                                col: [audio_df[col].values[0]] * len(df) if len(audio_df) > 0 else [0] * len(df)
                                for col in audio_feature_cols
                            })
                    
                    # Apply normalization if scalers are available
                    if self.feature_scalers:
                        for col in audio_feature_cols:
                            if col in self.feature_scalers:
                                feature_values = audio_features_df[col].values.reshape(-1, 1)
                                
                                # Fit or transform
                                if self.is_train:
                                    normalized_values = self.feature_scalers[col].fit_transform(feature_values)
                                else:
                                    normalized_values = self.feature_scalers[col].transform(feature_values)
                                
                                audio_features_df[col] = normalized_values.flatten()
                    
                    # Convert to numpy array
                    audio_features = audio_features_df.values
            except Exception as e:
                print(f"Error loading audio features for {pid}: {e}")
                audio_features = None
        
        # Normalize visual features if scalers are available
        if self.feature_scalers:
            normalized_features = features_df.copy()
            
            # Apply scaling for each feature separately
            for feature in features_df.columns:  # Only scale features that exist in the DataFrame
                if feature in self.feature_scalers:
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
        
        # Handle sequence length for visual features
        if features.shape[0] > self.max_seq_length:
            # Truncate sequence
            features = features[:self.max_seq_length]
            if audio_features is not None:
                audio_features = audio_features[:self.max_seq_length]
        elif features.shape[0] < self.max_seq_length:
            # Pad sequence
            visual_padding = np.zeros((self.max_seq_length - features.shape[0], features.shape[1]))
            features = np.vstack((features, visual_padding))
            
            # Also pad audio features if they exist
            if audio_features is not None:
                audio_padding = np.zeros((self.max_seq_length - audio_features.shape[0], audio_features.shape[1]))
                audio_features = np.vstack((audio_features, audio_padding))
        
        # Create audio placeholder if needed but not available
        if self.include_audio and audio_features is None:
            audio_features = np.zeros((self.max_seq_length, len(self.audio_feature_names)))
        
        # Get binary depression label
        binary_label = self.binary_labels[pid]
        
        return_dict = {
            'features': torch.FloatTensor(features),
            'binary_label': torch.FloatTensor([binary_label]),
            'participant_id': pid,
            'seq_length': min(features.shape[0], self.max_seq_length)
        }
        
        # Add audio features if included
        if self.include_audio:
            return_dict['audio_features'] = torch.FloatTensor(audio_features)
        
        # Add PHQ score if available (for reference only)
        if pid in self.phq_scores:
            return_dict['phq_score'] = torch.FloatTensor([self.phq_scores[pid]])
            
        return return_dict
    
    def get_feature_names(self):
        """Get feature names for interpretability"""
        return self.feature_names
    
    def get_audio_feature_names(self):
        """Get audio feature names for interpretability"""
        return self.audio_feature_names
    
    def get_all_feature_names(self):
        """Get all feature names (visual + audio)"""
        return self.feature_names + self.audio_feature_names
    
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
                  include_pose=True, include_audio=True):
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
        include_audio (bool): Whether to include audio features
        
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
        include_audio=include_audio
    )
    
    # Pre-fit the scalers on all training data
    print("Fitting feature normalizers on training data...")
    all_feature_data = {}
    
    # Initialize for all feature types (visual and audio)
    all_feature_names = train_dataset.get_feature_names()
    all_audio_names = []
    if include_audio:
        all_audio_names = train_dataset.get_audio_feature_names()
    
    # Initialize data collector for visual features
    for feature in all_feature_names:
        all_feature_data[feature] = []
    
    # Initialize data collector for audio features
    for feature in all_audio_names:
        all_feature_data[feature] = []
    
    # Collect feature values across all training participants
    for idx in range(len(train_dataset)):
        try:
            sample = train_dataset[idx]
            features = sample['features'].numpy()
            seq_length = sample['seq_length']
            
            # Only use valid sequence length (not padding)
            valid_features = features[:seq_length]
            
            # Add to collection for each visual feature
            for i, feature_name in enumerate(train_dataset.get_feature_names()):
                if i < valid_features.shape[1]:  # Make sure index is valid
                    all_feature_data[feature_name].extend(valid_features[:, i].tolist())
            
            # Process audio features if included
            if include_audio and 'audio_features' in sample:
                audio_features = sample['audio_features'].numpy()
                valid_audio = audio_features[:seq_length]
                
                # Add to collection for each audio feature
                for i, feature_name in enumerate(train_dataset.get_audio_feature_names()):
                    if i < valid_audio.shape[1]:  # Make sure index is valid
                        all_feature_data[feature_name].extend(valid_audio[:, i].tolist())
        except Exception as e:
            print(f"Error processing sample {idx} for feature normalization: {e}")
    
    # Fit each scaler on collected data
    feature_scalers = {}
    for feature in list(all_feature_names) + list(all_audio_names):
        if feature in all_feature_data and len(all_feature_data[feature]) > 0:
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
        include_audio=include_audio
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
        include_audio=include_audio
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
        include_audio=include_audio
    )
    
    # Print dataset statistics
    print("Dataset statistics:")
    for split, dataset in [('Train', train_dataset), ('Dev', dev_dataset), ('Test', test_dataset)]:
        dist = dataset.get_class_distribution()
        print(f"  {split}: {dist['total']} samples "
              f"({dist['depressed']} depressed, {dist['non_depressed']} non-depressed, "
              f"{dist['depressed']/dist['total']*100:.1f}% depressed)")
        
        if include_audio:
            print(f"  {split} features: {len(dataset.get_feature_names())} visual features, "
                  f"{len(dataset.get_audio_feature_names())} audio features")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train': {'loader': train_loader, 'dataset': train_dataset},
        'dev': {'loader': dev_loader, 'dataset': dev_dataset},
        'test': {'loader': test_loader, 'dataset': test_dataset}
    }
