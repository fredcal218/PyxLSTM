"""
Utility functions for depression severity prediction.

This module provides utility functions for the depression prediction project,
including seed setting, feature dimension calculation, and model evaluation.
"""

import os
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_feature_dimension(data_dir):
    """
    Get the dimension of the input features.
    
    Args:
        data_dir (str): Directory containing the E-DAIC data.
    
    Returns:
        int: Feature dimension.
    """
    # Find the first participant in the training set
    train_dir = os.path.join(data_dir, "data_extr", "train")
    participants = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    if not participants:
        raise ValueError("No participants found in the training directory")
    
    # Get the first participant's features path
    participant_id = participants[0].split("_")[0]
    features_path = os.path.join(train_dir, f"{participant_id}_P", "features",
                                f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")
    
    # Load the CSV and count the feature columns, excluding AU confidence columns
    df = pd.read_csv(features_path)
    feature_cols = [col for col in df.columns if 
                   (any(prefix in col for prefix in ['pose', 'gaze', 'AU']) and 
                    not col.endswith('_c'))]
    
    return len(feature_cols)


def plot_predictions(true_values, predictions, output_path):
    """
    Plot true vs predicted PHQ-8 scores.
    
    Args:
        true_values (list): True PHQ-8 scores.
        predictions (list): Predicted PHQ-8 scores.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predictions, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)
    
    plt.title(f'True vs Predicted PHQ-8 Scores\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}')
    plt.xlabel('True PHQ-8 Score')
    plt.ylabel('Predicted PHQ-8 Score')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()


def compute_confusion_matrix(true_values, predictions, threshold=10):
    """
    Compute confusion matrix for depression classification.
    
    Args:
        true_values (list): True PHQ-8 scores.
        predictions (list): Predicted PHQ-8 scores.
        threshold (int, optional): Threshold for depression (PHQ-8 >= threshold).
    
    Returns:
        dict: Confusion matrix metrics.
    """
    # Convert scores to binary classification (depressed or not)
    true_binary = [1 if score >= threshold else 0 for score in true_values]
    pred_binary = [1 if score >= threshold else 0 for score in predictions]
    
    # Calculate true positives, false positives, true negatives, false negatives
    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
