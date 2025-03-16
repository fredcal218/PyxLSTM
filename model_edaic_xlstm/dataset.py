import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class EDAICDataset(Dataset):
    def __init__(self, base_dir, split='train', seq_length=150, stride=50, transform=None, frame_step=10):
        """
        Dataset for E-DAIC data using the predefined split structure.

        Args:
            base_dir (str): Base directory containing the E-DAIC dataset
            split (str): Which split to use ('train', 'dev', or 'test')
            seq_length (int): Length of sequences to sample
            stride (int): Stride between consecutive sequences
            transform (callable, optional): Optional transform to be applied on samples
            frame_step (int): Sample every nth frame (default: 10)
        """
        self.base_dir = base_dir
        self.split = split
        self.seq_length = seq_length
        self.stride = stride
        self.transform = transform
        self.frame_step = frame_step  # Sample every nth frame

        self.data_dir = os.path.join(base_dir, 'data_extr', split)
        self.labels_path = os.path.join(base_dir, 'labels', f'{split}_split.csv')

        self.samples = []
        self.labels = []

        # Load depression scores from the labels CSV
        self.depression_scores = self._load_depression_scores()

        # Get all participant folders
        participant_folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f)) and '_P' in f]

        # Process each participant's data
        for folder in participant_folders:
            participant_id = folder.split('_')[0]
            self._process_participant(participant_id, folder)

        print(f"Loaded {split} set with {len(self.samples)} sequences from {len(participant_folders)} participants")
        print(f"Using frame step {frame_step} (1/{frame_step} of original frames)")

    def _load_depression_scores(self):
        """Load depression scores from the dataset labels CSV"""
        depression_scores = {}

        if os.path.exists(self.labels_path):
            df = pd.read_csv(self.labels_path)
            for _, row in df.iterrows():
                # Use PHQ_Score as the depression level label
                participant_id = str(row['Participant_ID'])
                phq_score = row['PHQ_Score']
                depression_scores[participant_id] = phq_score
        else:
            print(f"Warning: Labels file {self.labels_path} not found")

        return depression_scores

    def _process_participant(self, participant_id, folder):
        """Process a single participant's data into sequences with adaptive stride based on PHQ score"""
        # Path to the features CSV file
        file_path = os.path.join(self.data_dir, folder, 'features',
                               f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")

        if not os.path.exists(file_path):
            print(f"Warning: Features file {file_path} not found")
            return

        # Check if we have depression score for this participant
        if participant_id not in self.depression_scores:
            print(f"Warning: No depression score found for participant {participant_id}")
            return

        # Get participant's PHQ score
        phq_score = self.depression_scores[participant_id]
        
        # Determine stride based on depression severity
        # Use smaller stride for higher severity to generate more sequences
        if (phq_score >= 20):  # Severe (20-24)
            effective_stride = max(1, self.stride // 5)  # Generate ~5x more sequences
        elif (phq_score >= 15):  # Moderately severe (15-19)
            effective_stride = max(1, self.stride // 3)  # Generate ~3x more sequences
        elif (phq_score >= 10):  # Moderate (10-14)
            effective_stride = max(1, self.stride // 2)  # Generate ~2x more sequences
        else:
            effective_stride = self.stride  # Normal stride for mild/minimal
        
        # Load data
        try:
            df = pd.read_csv(file_path)
            
            # Sample only 1/n of the frames
            df = df.iloc[::self.frame_step].reset_index(drop=True)
            
            # Use adaptive sequence length
            effective_seq_length = self.seq_length

            # Filter feature columns
            # Keep only pose, gaze, and AU intensity features
            # Exclude AU confidence columns (columns with "_c" suffix)
            feature_cols = []
            
            # Add pose and gaze features
            pose_gaze_cols = [col for col in df.columns 
                             if any(x in col for x in ['pose', 'gaze']) 
                             and not col.startswith('frame') 
                             and not col.startswith('timestamp')]
            feature_cols.extend(pose_gaze_cols)
            
            # Add only AU intensity features (exclude AU confidence)
            au_intensity_cols = [col for col in df.columns 
                               if 'AU' in col and not col.endswith('_c')]
            feature_cols.extend(au_intensity_cols)
            
            # Extract selected features
            features = df[feature_cols].values
            
            # Print feature info on first participant
            if len(self.samples) == 0:
                print(f"Selected {len(feature_cols)} features:")
                print(f"  - Pose/gaze features: {len(pose_gaze_cols)}")
                print(f"  - AU intensity features: {len(au_intensity_cols)}")
                print(f"  - Excluded AU confidence features: {len([c for c in df.columns if 'AU' in c and c.endswith('_c')])}")
                print(f"  - Original frame count: {len(df) * self.frame_step} → Downsampled: {len(df)}")
                print(f"  - Using adaptive stride: Minimal/Mild={self.stride}, Moderate={self.stride//2}, "
                      f"Mod.Severe={self.stride//3}, Severe={self.stride//5}")

            # Handle missing values - replace NaN with 0
            features = np.nan_to_num(features, nan=0.0)

            # Create sequences using adaptive stride (more sequences for higher PHQ scores)
            for i in range(0, max(1, len(features) - effective_seq_length + 1), effective_stride):
                seq = features[i:i+effective_seq_length]
                
                # Pad sequence if needed (for last incomplete sequence)
                if len(seq) < effective_seq_length:
                    padding = np.zeros((effective_seq_length - len(seq), seq.shape[1]))
                    seq = np.vstack([seq, padding])
                    
                self.samples.append(seq)

                # Add depression score as label
                self.labels.append(phq_score)
        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        # Convert to tensors
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
