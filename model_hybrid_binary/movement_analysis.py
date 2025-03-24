import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal

class MovementFeatureExtractor:
    """
    Extracts movement patterns from pose features, transforming positional data
    into velocity, acceleration, and directional change features.
    """
    def __init__(self, pose_features=None, smoothing_window=5):
        """
        Initialize the movement feature extractor
        
        Args:
            pose_features (list): List of pose feature names to analyze (default: all pose_* features)
            smoothing_window (int): Window size for smoothing position data before derivative calculation
        """
        self.pose_features = pose_features or [
            'pose_Tx', 'pose_Ty', 'pose_Tz',  # Position
            'pose_Rx', 'pose_Ry', 'pose_Rz'   # Rotation
        ]
        self.smoothing_window = smoothing_window
        
        # Create derived feature names
        self.velocity_features = [f"{f}_velocity" for f in self.pose_features]
        self.acceleration_features = [f"{f}_acceleration" for f in self.pose_features]
        self.direction_features = [f"{f}_direction_changes" for f in self.pose_features]
        
        # All derived features combined
        self.all_derived_features = (
            self.velocity_features + 
            self.acceleration_features + 
            self.direction_features
        )
    
    def extract_movement_features(self, df):
        """
        Extract movement features from a dataframe with pose features
        
        Args:
            df (pd.DataFrame): DataFrame containing pose features
            
        Returns:
            pd.DataFrame: DataFrame with original features plus derived movement features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Only process features that exist in the dataframe
        available_pose_features = [f for f in self.pose_features if f in df.columns]
        
        if not available_pose_features:
            print("Warning: No pose features found in the input data")
            return result_df
        
        # Process each pose feature
        for feature in available_pose_features:
            # Get the raw position/rotation data
            raw_values = df[feature].values
            
            # Apply smoothing filter to reduce noise (important before derivatives)
            smooth_values = self._smooth_signal(raw_values)
            
            # Calculate first derivative (velocity)
            velocity = self._calculate_derivative(smooth_values)
            velocity_feature = f"{feature}_velocity"
            result_df[velocity_feature] = velocity
            
            # Calculate second derivative (acceleration)
            acceleration = self._calculate_derivative(velocity)
            acceleration_feature = f"{feature}_acceleration"
            result_df[acceleration_feature] = acceleration
            
            # Calculate direction changes
            direction_changes = self._calculate_direction_changes(velocity)
            direction_feature = f"{feature}_direction_changes"
            result_df[direction_feature] = direction_changes
        
        return result_df
    
    def _smooth_signal(self, values):
        """Apply smoothing to reduce noise before derivative calculation"""
        if len(values) < self.smoothing_window:
            return values
        
        return signal.savgol_filter(
            values, 
            window_length=self.smoothing_window,
            polyorder=2,
            mode='nearest'
        )
    
    def _calculate_derivative(self, values):
        """Calculate derivative (difference between consecutive points)"""
        # Pad with zeros to maintain same length
        derivatives = np.zeros_like(values)
        
        # Calculate differences for all but first element
        derivatives[1:] = np.diff(values)
        
        return derivatives
    
    def _calculate_direction_changes(self, velocity):
        """Calculate points where direction changes (sign of velocity changes)"""
        # Initialize with zeros
        direction_changes = np.zeros_like(velocity)
        
        # Check for sign changes in velocity
        for i in range(1, len(velocity)):
            if (velocity[i] > 0 and velocity[i-1] <= 0) or \
               (velocity[i] < 0 and velocity[i-1] >= 0):
                direction_changes[i] = 1
        
        # Create a cumulative sum to represent total direction changes up to each point
        cumulative_changes = np.cumsum(direction_changes)
        
        return cumulative_changes
    
    def calculate_movement_statistics(self, df):
        """
        Calculate summary statistics for movement features
        
        Args:
            df (pd.DataFrame): DataFrame with movement features already extracted
            
        Returns:
            dict: Dictionary of movement statistics
        """
        stats = {}
        
        # Calculate statistics for each feature type
        for feature in self.pose_features:
            if feature not in df.columns:
                continue
                
            velocity_feature = f"{feature}_velocity"
            if velocity_feature in df.columns:
                # Velocity statistics
                velocity_values = df[velocity_feature].values
                stats[f"{velocity_feature}_mean"] = np.mean(np.abs(velocity_values))
                stats[f"{velocity_feature}_std"] = np.std(velocity_values)
                stats[f"{velocity_feature}_max"] = np.max(np.abs(velocity_values))
            
            acceleration_feature = f"{feature}_acceleration"
            if acceleration_feature in df.columns:
                # Acceleration statistics
                acceleration_values = df[acceleration_feature].values
                stats[f"{acceleration_feature}_mean"] = np.mean(np.abs(acceleration_values))
                stats[f"{acceleration_feature}_std"] = np.std(acceleration_values)
                stats[f"{acceleration_feature}_max"] = np.max(np.abs(acceleration_values))
            
            direction_feature = f"{feature}_direction_changes"
            if direction_feature in df.columns:
                # Direction change statistics
                direction_values = df[direction_feature].values
                if len(direction_values) > 0:
                    stats[f"{direction_feature}_total"] = direction_values[-1]
                    stats[f"{direction_feature}_rate"] = direction_values[-1] / len(direction_values)
        
        return stats
    
    def visualize_movement(self, df, feature, output_path=None):
        """
        Visualize original position and derived movement features
        
        Args:
            df (pd.DataFrame): DataFrame with extracted movement features
            feature (str): Base feature name (e.g., 'pose_Tx')
            output_path (str): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Movement visualization
        """
        velocity_feature = f"{feature}_velocity"
        acceleration_feature = f"{feature}_acceleration"
        direction_feature = f"{feature}_direction_changes"
        
        if feature not in df.columns or velocity_feature not in df.columns:
            print(f"Feature {feature} or its derivatives not found in data")
            return None
        
        # Create visualization
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Position
        axs[0].plot(df[feature], label=feature)
        axs[0].set_title(f'Position: {feature}')
        axs[0].set_ylabel('Position Value')
        axs[0].grid(True)
        
        # Velocity
        axs[1].plot(df[velocity_feature], label=velocity_feature, color='orange')
        axs[1].set_title(f'Velocity (first derivative)')
        axs[1].set_ylabel('Velocity')
        axs[1].grid(True)
        
        # Acceleration
        if acceleration_feature in df.columns:
            axs[2].plot(df[acceleration_feature], label=acceleration_feature, color='green')
            axs[2].set_title(f'Acceleration (second derivative)')
            axs[2].set_ylabel('Acceleration')
            axs[2].grid(True)
        
        # Direction changes
        if direction_feature in df.columns:
            axs[3].plot(df[direction_feature], label=direction_feature, color='red')
            axs[3].set_title(f'Direction Changes (cumulative)')
            axs[3].set_xlabel('Frame Index')
            axs[3].set_ylabel('Direction Changes')
            axs[3].grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig

def analyze_participant_movement(participant_df, output_dir=None):
    """
    Analyze movement patterns for a single participant
    
    Args:
        participant_df (pd.DataFrame): DataFrame with participant feature data
        output_dir (str): Directory to save visualizations
        
    Returns:
        tuple: (DataFrame with movement features, movement statistics dict)
    """
    # Create movement extractor
    extractor = MovementFeatureExtractor()
    
    # Extract movement features
    movement_df = extractor.extract_movement_features(participant_df)
    
    # Calculate statistics
    movement_stats = extractor.calculate_movement_statistics(movement_df)
    
    # Create visualizations if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize each pose feature
        for feature in extractor.pose_features:
            if feature in participant_df.columns:
                output_path = os.path.join(output_dir, f"{feature}_movement.png")
                extractor.visualize_movement(movement_df, feature, output_path)
    
    return movement_df, movement_stats
