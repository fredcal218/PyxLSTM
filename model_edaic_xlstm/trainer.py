import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_fscore_support
import os
import time
import math
import json
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add WeightedMAELoss class directly to trainer.py
class WeightedMAELoss(nn.Module):
    """
    Weighted Mean Absolute Error loss that gives higher importance to higher PHQ-8 scores.
    This helps address class imbalance by making errors on underrepresented higher severity
    depression scores more costly.
    """
    def __init__(self, scale_factor=0.2):
        """
        Initialize weighted MAE loss.
        
        Args:
            scale_factor (float): Controls how much weight increases with target value.
                                 Higher values give more emphasis to high PHQ-8 scores.
        """
        super(WeightedMAELoss, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, predictions, targets):
        """
        Calculate weighted MAE loss.
        
        Args:
            predictions (torch.Tensor): Model predictions (B, 1)
            targets (torch.Tensor): Ground truth values (B, 1)
            
        Returns:
            torch.Tensor: Weighted MAE loss
        """
        # Base loss is absolute error (L1)
        mae = torch.abs(predictions - targets)
        
        # Calculate weights based on target values
        # Higher targets (more severe depression) get higher weights
        weights = 1.0 + self.scale_factor * targets
        
        # Apply weights to the loss
        weighted_mae = mae * weights
        
        # Return mean loss
        return weighted_mae.mean()

# Add new CombinedLoss class below WeightedMAELoss
class CombinedLoss(nn.Module):
    """
    Combined loss function that incorporates both weighting for higher PHQ-8 scores
    and variance regularization to encourage wider prediction ranges.
    """
    def __init__(self, scale_factor=0.3, min_variance=16.0, lambda_var=0.5):
        """
        Initialize combined loss function.
        
        Args:
            scale_factor (float): Weight scale factor for higher PHQ-8 scores
            min_variance (float): Minimum desired variance for predictions
            lambda_var (float): Weight of the variance regularization term
        """
        super(CombinedLoss, self).__init__()
        self.scale_factor = scale_factor
        self.min_variance = min_variance
        self.lambda_var = lambda_var
        
    def forward(self, predictions, targets):
        """
        Calculate combined loss with weighted MAE and variance regularization.
        
        Args:
            predictions (torch.Tensor): Model predictions (B, 1)
            targets (torch.Tensor): Ground truth values (B, 1)
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Base loss is absolute error (L1)
        mae = torch.abs(predictions - targets)
        
        # Calculate weights based on target values
        # Higher targets (more severe depression) get higher weights
        weights = 1.0 + self.scale_factor * targets
        
        # Apply weights to the loss
        weighted_mae = mae * weights
        weighted_mae = weighted_mae.mean()
        
        # Calculate variance of predictions
        variance = torch.var(predictions)
        
        # Penalty if variance is too low
        variance_penalty = torch.max(torch.tensor(0.0, device=predictions.device), 
                                    self.min_variance - variance)
        
        # Combined loss with both weighting and variance regularization
        total_loss = weighted_mae + self.lambda_var * variance_penalty
        
        return total_loss

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, learning_rate, save_dir, criterion=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Use custom criterion if provided, otherwise default to MAE loss
        self.criterion = criterion if criterion is not None else nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Add learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,  # Reduce LR by half when plateauing
            patience=10,  # Wait for 15 epochs before reducing LR
            verbose=True, 
            min_lr=1e-8
        )

        # Create stratified sampler for training data
        train_sampler = self.get_balanced_sampler(self.train_dataset)
        
        # DataLoader with stratified sampling for training
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler
        )
        
        # Use regular DataLoader for validation (no need for stratification)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        
        print("Using stratified sampling for training data to balance depression severity levels")
    
    def get_balanced_sampler(self, dataset):
        """
        Create a sampler that over-samples from minority classes and under-samples from majority classes.
        
        Args:
            dataset: The dataset to sample from
            
        Returns:
            WeightedRandomSampler: Sampler that balances classes
        """
        # Extract labels
        labels = np.array(dataset.labels)
        
        # Convert PHQ-8 scores to severity levels (0-4)
        # 0-4: None/minimal, 5-9: Mild, 10-14: Moderate, 15-19: Moderately severe, 20-24: Severe
        severity_levels = np.digitize(labels, bins=[0, 5, 10, 15, 20, 25]) - 1
        
        # Count samples in each severity level
        level_counts = np.bincount(severity_levels)
        
        # Print class distribution for information
        level_names = ['None/Minimal', 'Mild', 'Moderate', 'Mod. Severe', 'Severe']
        print("Class distribution before sampling:")
        for i, name in enumerate(level_names):
            if i < len(level_counts):
                print(f"  {name}: {level_counts[i]} samples ({level_counts[i]/len(severity_levels)*100:.1f}%)")
            else:
                print(f"  {name}: 0 samples (0.0%)")
        
        # Calculate weights (inverse of frequency)
        # We add a small epsilon to prevent division by zero
        weights = 1.0 / (level_counts[severity_levels] + 1e-8)
        
        # Create and return sampler
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def visualize_metrics(self, history, best_epoch):
        """
        Create and save visualization plots for training and validation metrics.
        
        Args:
            history (dict): Dictionary containing training history metrics
            best_epoch (int): The epoch with the best validation loss (1-indexed)
        """
        # Convert to 0-indexed for array indexing
        best_epoch_idx = best_epoch - 1
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Set figure style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(self.save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Ensure all tensors are moved to CPU before plotting
        history_cpu = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_cpu[key] = [v.cpu() if torch.is_tensor(v) else v for v in value]
            else:
                history_cpu[key] = value.cpu() if torch.is_tensor(value) else value
        
        # Plot 1: Regression Metrics (RMSE, MAE, Loss)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(epochs, history_cpu['train_rmse'], label='Train')
        plt.plot(epochs, history_cpu['val_rmse'], label='Validation')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
        plt.title('RMSE over Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(epochs, history_cpu['train_mae'], label='Train')
        plt.plot(epochs, history_cpu['val_mae'], label='Validation')
        plt.axvline(x=best_epoch, color='r', linestyle='--')
        plt.title('MAE over Epochs')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(epochs, history_cpu['train_loss'], label='Train')
        plt.plot(epochs, history_cpu['val_loss'], label='Validation')
        plt.axvline(x=best_epoch, color='r', linestyle='--')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'regression_metrics.png'), dpi=300)
        plt.close()
        
        # Plot 2: Accuracy Metrics
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(epochs, history_cpu['train_binary_acc'], label='Train')
        plt.plot(epochs, history_cpu['val_binary_acc'], label='Validation')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
        plt.title('Binary Accuracy over Epochs (PHQ > 10)')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(epochs, history_cpu['train_overall_acc'], label='Train')
        plt.plot(epochs, history_cpu['val_overall_acc'], label='Validation')
        plt.axvline(x=best_epoch, color='r', linestyle='--')
        plt.title('Overall Accuracy over Epochs (Tolerance = ±1)')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(epochs, history_cpu['train_level_acc'], label='Train')
        plt.plot(epochs, history_cpu['val_level_acc'], label='Validation')
        plt.axvline(x=best_epoch, color='r', linestyle='--')
        plt.title('PHQ Level Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'accuracy_metrics.png'), dpi=300)
        plt.close()
        
        # Plot 3: Per-Level Accuracies
        plt.figure(figsize=(12, 10))
        level_names = ['None/Minimal', 'Mild', 'Moderate', 'Mod. Severe', 'Severe']
        
        for i, level in enumerate(level_names):
            plt.subplot(len(level_names), 1, i+1)
            plt.plot(epochs, history_cpu[f'val_level{i}_acc'], label=f'Validation')
            plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
            plt.title(f'{level} Accuracy over Epochs')
            plt.ylabel('Accuracy')
            if i == 0:
                plt.legend()
            if i == len(level_names) - 1:
                plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'per_level_accuracies.png'), dpi=300)
        plt.close()
        
        # Plot 4: Threshold Accuracies
        plt.figure(figsize=(12, 8))
        threshold_values = [5, 10, 15, 20]
        
        for i, threshold in enumerate(threshold_values):
            plt.plot(epochs, history_cpu[f'val_threshold_{threshold}_acc'], 
                     label=f'T{threshold}', 
                     marker='o', 
                     markersize=4)
        
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
        plt.title('Threshold Accuracies over Epochs (±2 points near threshold)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'threshold_accuracies.png'), dpi=300)
        plt.close()
        
        print(f"Saved visualization plots to {plots_dir}")

    def train(self, epochs, patience, seed):
        best_val_mae = float('inf')  # Track MAE instead of loss for early stopping
        patience_counter = 0
        best_epoch = 0
        
        # Update history to include per-level accuracies and score tolerance metrics
        history = {
            'train_loss': [], 'train_binary_acc': [], 'train_overall_acc': [], 'train_level_acc': [],
            'train_level0_acc': [], 'train_level1_acc': [], 'train_level2_acc': [], 
            'train_level3_acc': [], 'train_level4_acc': [],
            'train_score_tolerance_acc': [], 
            'train_threshold_5_acc': [], 'train_threshold_10_acc': [],
            'train_threshold_15_acc': [], 'train_threshold_20_acc': [],
            'train_rmse': [], 'train_mae': [], 'train_r2': [],
            'val_loss': [], 'val_binary_acc': [], 'val_overall_acc': [], 'val_level_acc': [],
            'val_level0_acc': [], 'val_level1_acc': [], 'val_level2_acc': [], 
            'val_level3_acc': [], 'val_level4_acc': [],
            'val_score_tolerance_acc': [],
            'val_threshold_5_acc': [], 'val_threshold_10_acc': [],
            'val_threshold_15_acc': [], 'val_threshold_20_acc': [],
            'val_rmse': [], 'val_mae': [], 'val_r2': [],
            'learning_rates': [],
            'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
        }

        print(f"Starting training on device: {self.device}")
        print(f"Initial learning rate: {self.learning_rate}")
        print(f"Using L1Loss (MAE) as criterion")
        
        for epoch in range(epochs):
            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            train_metrics = train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.device)
            val_metrics = validate_epoch(self.model, self.val_loader, self.criterion, self.device)

            # Unpack metrics (updated to include score tolerance metrics)
            train_loss, train_binary_acc, train_overall_acc, (train_level_acc, train_level_accs), \
                (train_score_tol_acc, train_threshold_accs), train_rmse, train_mae, train_r2, \
                train_precision, train_recall, train_f1 = train_metrics
            
            val_loss, val_binary_acc, val_overall_acc, (val_level_acc, val_level_accs), \
                (val_score_tol_acc, val_threshold_accs), val_rmse, val_mae, val_r2, \
                val_precision, val_recall, val_f1 = val_metrics
            
            # Update learning rate scheduler based on validation MAE
            self.scheduler.step(val_mae)
            
            # Store metrics in history (updated for score tolerance metrics)
            history['train_loss'].append(train_loss)
            history['train_binary_acc'].append(train_binary_acc)
            history['train_overall_acc'].append(train_overall_acc)
            history['train_level_acc'].append(train_level_acc)
            for i, acc in enumerate(train_level_accs):
                history[f'train_level{i}_acc'].append(acc)
            history['train_score_tolerance_acc'].append(train_score_tol_acc)
            threshold_names = ["Minimal/Mild (5)", "Mild/Moderate (10)", 
                              "Moderate/Mod.Severe (15)", "Mod.Severe/Severe (20)"]
            for i, name in enumerate(threshold_names):
                history[f'train_threshold_{[5, 10, 15, 20][i]}_acc'].append(train_threshold_accs.get(name, 0.0))
            history['train_rmse'].append(train_rmse)
            history['train_mae'].append(train_mae)
            history['train_r2'].append(train_r2)
            history['val_loss'].append(val_loss)
            history['val_binary_acc'].append(val_binary_acc)
            history['val_overall_acc'].append(val_overall_acc)
            history['val_level_acc'].append(val_level_acc)
            for i, acc in enumerate(val_level_accs):
                history[f'val_level{i}_acc'].append(acc)
            history['val_score_tolerance_acc'].append(val_score_tol_acc)
            for i, name in enumerate(threshold_names):
                history[f'val_threshold_{[5, 10, 15, 20][i]}_acc'].append(val_threshold_accs.get(name, 0.0))
            history['val_rmse'].append(val_rmse)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_f1'].append(train_f1)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)

            # Print metrics with all threshold metrics shown
            print(f'Epoch [{epoch + 1}/{epochs}]: (LR: {current_lr:.6f})')
            print(f'  Train - Loss (MAE): {train_loss:.4f}, Binary: {train_binary_acc:.4f}, Overall: {train_overall_acc:.4f}')
            print(f'          Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
            print(f'          Score-Tol: {train_score_tol_acc:.4f} (±2 pts)')
            print(f'          Thresholds: [T5: {train_threshold_accs.get("Minimal/Mild (5)", 0.0):.4f}, ' +
                  f'T10: {train_threshold_accs.get("Mild/Moderate (10)", 0.0):.4f}, ' +
                  f'T15: {train_threshold_accs.get("Moderate/Mod.Severe (15)", 0.0):.4f}, ' +
                  f'T20: {train_threshold_accs.get("Mod.Severe/Severe (20)", 0.0):.4f}]')
            print(f'          Per-level: {train_level_acc:.4f} (exact) [None: {train_level_accs[0]:.4f}, Mild: {train_level_accs[1]:.4f}, Mod: {train_level_accs[2]:.4f}, Sev+: {train_level_accs[3]:.4f}, Severe: {train_level_accs[4]:.4f}]')
            print(f'          RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}')
            print(f'  Val   - Loss (MAE): {val_loss:.4f}, Binary: {val_binary_acc:.4f}, Overall: {val_overall_acc:.4f}')
            print(f'          Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
            print(f'          Score-Tol: {val_score_tol_acc:.4f} (±2 pts)')
            print(f'          Thresholds: [T5: {val_threshold_accs.get("Minimal/Mild (5)", 0.0): .4f}, ' +
                  f'T10: {val_threshold_accs.get("Mild/Moderate (10)", 0.0):.4f}, ' +
                  f'T15: {val_threshold_accs.get("Moderate/Mod.Severe (15)", 0.0):.4f}, ' +
                  f'T20: {val_threshold_accs.get("Mod.Severe/Severe (20)", 0.0):.4f}]')
            print(f'          Per-level: {val_level_acc:.4f} (exact) [None: {val_level_accs[0]:.4f}, Mild: {val_level_accs[1]:.4f}, Mod: {val_level_accs[2]:.4f}, Sev+: {val_level_accs[3]:.4f}, Severe: {val_level_accs[4]:.4f}]')
            print(f'          RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}')

            # Use MAE for early stopping instead of loss
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"  New best model! Saving checkpoint (val_mae: {val_mae:.4f})")
                self.save_model()
                
                # Create and save visualization plots for current best model
                self.visualize_metrics(history, best_epoch)
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter}/{patience} epochs (best val_mae: {best_val_mae:.4f} @ epoch {best_epoch})")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                print(f"Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
                
                # Final visualization after early stopping
                self.visualize_metrics(history, best_epoch)
                break

        # Final visualization if no early stopping occurred
        if patience_counter < patience:
            self.visualize_metrics(history, best_epoch)

        # Save training history
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            history_serializable = {
                k: [float(val) for val in v]
                for k, v in history.items()
            }
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        print(f"Training completed after {len(history['train_loss'])} epochs")
        print(f"Best model was at epoch {best_epoch} with validation MAE: {best_val_mae:.4f}")
        
        return history

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        model_path = os.path.join(self.save_dir, 'best_model.pt')
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_params': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'device': str(self.device),
                'criterion': str(self.criterion),
                'optimizer': str(self.optimizer),
                'scheduler': str(self.scheduler),
                'model_structure': str(self.model),
                'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }, model_path)
        
        print(f'Model saved to {model_path}')
        
    def load_best_model(self):
        """Load the best model from the saved checkpoint"""
        model_path = os.path.join(self.save_dir, 'best_model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded best model from {model_path}")
            return True
        else:
            print(f"No saved model found at {model_path}")
            return False

def calculate_regression_metrics(predictions, targets):
    """Calculate common regression metrics"""
    # Convert tensors to numpy arrays for sklearn metrics
    y_pred = predictions.detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    # R² score (coefficient of determination)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, r2

def calculate_classification_metrics(predictions, targets, threshold=10.0):
    """
    Calculate binary classification metrics (precision, recall, F1).
    
    Args:
        predictions (torch.Tensor): Model's predicted depression scores
        targets (torch.Tensor): Ground truth depression scores
        threshold (float): Clinical threshold for depression (default: 10.0)
        
    Returns:
        tuple: (precision, recall, f1)
    """
    # Convert regression outputs to binary predictions
    binary_preds = (predictions.detach().cpu().numpy() > threshold).astype(int).flatten()
    binary_targets = (targets.detach().cpu().numpy() > threshold).astype(int).flatten()
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_targets, binary_preds, average='binary', zero_division=0)
    
    return precision, recall, f1

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_binary_acc = 0.0
    running_overall_acc = 0.0
    running_level_acc = 0.0
    all_predictions = []
    all_targets = []
    total_samples = 0
    
    for inputs, targets in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        # Calculate multiple accuracy metrics
        binary_acc = model.calculate_accuracy(outputs, targets)
        overall_acc = model.calculate_overall_accuracy(outputs, targets)
        level_acc_tuple = model.calculate_per_level_accuracy(outputs, targets)
        score_tolerance_tuple = model.calculate_score_tolerance_accuracy(outputs, targets, tolerance=2.0)
        
        # Store predictions and targets for metric calculation
        all_predictions.append(outputs.detach())
        all_targets.append(targets.detach())
        
        # Update metrics
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_binary_acc += binary_acc * batch_size
        running_overall_acc += overall_acc * batch_size
        running_level_acc += level_acc_tuple[0] * batch_size
        total_samples += batch_size
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate regression metrics
    rmse, mae, r2 = calculate_regression_metrics(predictions, targets)
    
    # Calculate binary classification metrics
    precision, recall, f1 = calculate_classification_metrics(predictions, targets, threshold=10.0)
    
    # Calculate final metrics
    epoch_loss = running_loss / total_samples
    epoch_binary_acc = running_binary_acc / total_samples
    epoch_overall_acc = running_overall_acc / total_samples
    epoch_level_acc_tuple = (running_level_acc / total_samples, [0.0, 0.0, 0.0, 0.0, 0.0])  # Initialize with zeros
    
    # Calculate per-level accuracies by running model.calculate_per_level_accuracy on all predictions
    if len(all_predictions) > 0 and len(all_targets) > 0:
        epoch_level_acc_tuple = model.calculate_per_level_accuracy(predictions, targets)
    
    # Calculate score tolerance accuracy on all predictions
    score_tolerance_tuple = model.calculate_score_tolerance_accuracy(predictions, targets, tolerance=2.0)
    
    return epoch_loss, epoch_binary_acc, epoch_overall_acc, epoch_level_acc_tuple, score_tolerance_tuple, rmse, mae, r2, precision, recall, f1

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_binary_acc = 0.0
    running_overall_acc = 0.0
    running_level_acc = 0.0
    all_predictions = []
    all_targets = []
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate multiple accuracy metrics
            binary_acc = model.calculate_accuracy(outputs, targets)
            overall_acc = model.calculate_overall_accuracy(outputs, targets)
            level_acc_tuple = model.calculate_per_level_accuracy(outputs, targets)
            score_tolerance_tuple = model.calculate_score_tolerance_accuracy(outputs, targets, tolerance=2.0)
            
            # Store predictions and targets for metric calculation
            all_predictions.append(outputs)
            all_targets.append(targets)
            
            # Update metrics
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_binary_acc += binary_acc * batch_size
            running_overall_acc += overall_acc * batch_size
            running_level_acc += level_acc_tuple[0] * batch_size
            total_samples += batch_size
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate regression metrics
    rmse, mae, r2 = calculate_regression_metrics(predictions, targets)
    
    # Calculate binary classification metrics
    precision, recall, f1 = calculate_classification_metrics(predictions, targets, threshold=10.0)
    
    # Calculate final metrics
    val_loss = running_loss / total_samples
    val_binary_acc = running_binary_acc / total_samples
    val_overall_acc = running_overall_acc / total_samples
    val_level_acc_tuple = (running_level_acc / total_samples, [0.0, 0.0, 0.0, 0.0, 0.0])  # Initialize with zeros
    
    # Calculate per-level accuracies by running model.calculate_per_level_accuracy on all predictions
    if len(all_predictions) > 0 and len(all_targets) > 0:
        val_level_acc_tuple = model.calculate_per_level_accuracy(predictions, targets)
    
    # Calculate score tolerance accuracy on all predictions
    score_tolerance_tuple = model.calculate_score_tolerance_accuracy(predictions, targets, tolerance=2.0)
    
    return val_loss, val_binary_acc, val_overall_acc, val_level_acc_tuple, score_tolerance_tuple, rmse, mae, r2, precision, recall, f1