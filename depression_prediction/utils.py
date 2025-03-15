"""
Utility functions for depression severity prediction.

This module provides utility functions for the depression prediction project,
including seed setting, feature dimension calculation, and model evaluation.
"""

import os
import gc  # Import at the top to avoid "gc is not defined" error
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
    # Clean NaN or Inf values before plotting
    valid_indices = []
    for i in range(len(true_values)):
        if (not np.isnan(true_values[i]) and not np.isnan(predictions[i]) and 
            not np.isinf(true_values[i]) and not np.isinf(predictions[i])):
            valid_indices.append(i)
    
    if not valid_indices:
        print("Warning: No valid prediction pairs to plot")
        return
        
    clean_true = [true_values[i] for i in valid_indices]
    clean_pred = [predictions[i] for i in valid_indices]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(clean_true, clean_pred, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(min(clean_true), min(clean_pred))
    max_val = max(max(clean_true), max(clean_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics
    mae = mean_absolute_error(clean_true, clean_pred)
    rmse = np.sqrt(mean_squared_error(clean_true, clean_pred))
    r2 = r2_score(clean_true, clean_pred)
    
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
    # Clean NaN or Inf values
    valid_indices = []
    for i in range(len(true_values)):
        if (not np.isnan(true_values[i]) and not np.isnan(predictions[i]) and 
            not np.isinf(true_values[i]) and not np.isinf(predictions[i])):
            valid_indices.append(i)
    
    if not valid_indices:
        print("Warning: No valid prediction pairs for confusion matrix")
        return {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }
        
    clean_true = [true_values[i] for i in valid_indices]
    clean_pred = [predictions[i] for i in valid_indices]
    
    # Convert scores to binary classification (depressed or not)
    true_binary = [1 if score >= threshold else 0 for score in clean_true]
    pred_binary = [1 if score >= threshold else 0 for score in clean_pred]
    
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


def print_memory_stats():
    """Print memory statistics for PyTorch tensors and CUDA if available."""
    print("\n=== Memory Statistics ===")
    
    # Report CUDA memory if available
    if torch.cuda.is_available():
        print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"CUDA max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA max reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
        
    # Report number of PyTorch tensors
    total_tensors = 0
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                total_tensors += 1
                total_size += obj.element_size() * obj.nelement()
        except:
            pass
    
    print(f"Total tensors: {total_tensors}")
    print(f"Total tensor memory: {total_size / 1024**2:.2f} MB")


def plot_metrics(metrics_dict, output_path):
    """
    Plot metrics as a bar chart.
    
    Args:
        metrics_dict (dict): Dictionary of metrics to plot (name -> value).
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Custom colors for different metrics
    colors = ['#4CAF50', '#FF5722', '#2196F3', '#FFC107'][:len(metrics)]
    
    plt.bar(metrics, values, color=colors)
    plt.title(f'Depression Prediction Metrics\n' + 
              '\n'.join([f"{m}: {v:.4f}" for m, v in metrics_dict.items()]))
    plt.ylabel('Value')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, val in enumerate(values):
        plt.text(i, val + max(values) * 0.01, f"{val:.4f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Metrics plot saved to {output_path}")


def plot_evaluation_results(true_values, predictions, output_dir, prefix=""):
    """
    Create and save comprehensive evaluation plots.
    
    Args:
        true_values (list): True PHQ-8 scores.
        predictions (list): Predicted PHQ-8 scores.
        output_dir (str): Directory to save plots.
        prefix (str, optional): Prefix for plot filenames.
    
    Returns:
        dict: Dictionary of calculated metrics.
    """
    # Clean data and calculate metrics
    valid_indices = []
    for i in range(len(true_values)):
        if (not np.isnan(true_values[i]) and not np.isnan(predictions[i]) and 
            not np.isinf(true_values[i]) and not np.isinf(predictions[i])):
            valid_indices.append(i)
    
    if not valid_indices:
        print("Warning: No valid prediction pairs to plot")
        return {
            "mae": float('nan'),
            "rmse": float('nan'),
            "r2": float('nan')
        }
        
    clean_true = np.array([true_values[i] for i in valid_indices])
    clean_pred = np.array([predictions[i] for i in valid_indices])
    
    # Calculate metrics
    mae = mean_absolute_error(clean_true, clean_pred)
    rmse = np.sqrt(mean_squared_error(clean_true, clean_pred))
    r2 = r2_score(clean_true, clean_pred)
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    
    # 1. Plot true vs predicted values
    plot_predictions(clean_true, clean_pred, 
                    os.path.join(output_dir, f"{prefix}predictions.png"))
    
    # 2. Plot metrics
    plot_metrics({"MAE": mae, "RMSE": rmse}, 
                os.path.join(output_dir, f"{prefix}metrics.png"))
    
    # 3. Plot error distribution
    errors = clean_pred - clean_true
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, alpha=0.7, color='#2196F3')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Prediction Error Distribution\nMean Error: {np.mean(errors):.4f}, Std Dev: {np.std(errors):.4f}')
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{prefix}error_distribution.png"))
    plt.close()
    
    print(f"Evaluation plots saved to {output_dir}")
    return metrics
