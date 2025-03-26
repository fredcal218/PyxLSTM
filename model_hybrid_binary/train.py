import os
import random  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
import json
import math

from model import HybridDepBinaryClassifier
from data_processor import get_dataloaders

# Set seed for reproducibility
SEED = 42

def set_seed(seed=SEED):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Call set_seed at the beginning
set_seed()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 2  # Layers for both CNN and xLSTM paths
DROPOUT = 0.5    # Increased from 0.4 for more regularization
POSE_DROPOUT = 0.9  # Increased from 0.6 to reduce pose feature dominance  
AUDIO_DROPOUT = 0.5  # Increased from 0.4 for more regularization
MAX_SEQ_LENGTH = 450
LEARNING_RATE = 0.00003  # Decreased from 0.0005 for more stable optimization
WEIGHT_DECAY = 0.02    # Increased from 0.01 for stronger regularization
NUM_EPOCHS = 150        # Increased to allow more time to find optimal weights
EARLY_STOPPING_PATIENCE = 20  # Increased from 15
EARLY_STOPPING_MIN_DELTA = 0.005  # Minimum improvement required
WARMUP_EPOCHS = 10       # Number of epochs for learning rate warm-up
LR_SCHEDULER_FACTOR = 0.7  # Less aggressive reduction (was 0.5) 
LR_SCHEDULER_PATIENCE = 7  # Increased patience before reducing LR
INCLUDE_MOVEMENT_FEATURES = True
INCLUDE_POSE = True
INCLUDE_AUDIO = True  
USE_HYBRID_PATHWAYS = True
# Fusion type parameter - options: "attention" (original), "cross_modal"
FUSION_TYPE = "cross_modal"

# Directories
DATA_DIR = "E-DAIC/data_extr"
LABELS_DIR = "E-DAIC/labels"
timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
OUTPUT_DIR = f"model_hybrid_binary/results/binary_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration logging
config = {
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'pose_dropout': POSE_DROPOUT,
    'audio_dropout': AUDIO_DROPOUT,
    'max_seq_length': MAX_SEQ_LENGTH,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'num_epochs': NUM_EPOCHS,
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA,
    'warmup_epochs': WARMUP_EPOCHS,
    'lr_scheduler_factor': LR_SCHEDULER_FACTOR,
    'lr_scheduler_patience': LR_SCHEDULER_PATIENCE,
    'include_movement_features': INCLUDE_MOVEMENT_FEATURES,
    'include_pose': INCLUDE_POSE,
    'include_audio': INCLUDE_AUDIO,
    'use_hybrid_pathways': USE_HYBRID_PATHWAYS,
    'fusion_type': FUSION_TYPE,
    'model_type': 'Hybrid CNN-xLSTM-Audio',
    'device': str(device)
}

# Save configuration
with open(os.path.join(OUTPUT_DIR, 'config.txt'), 'w') as f:
    for key, value in config.items():
        f.write(f"{key}: {value}\n")

def optimize_threshold(model, dev_loader):
    """Find optimal classification threshold that maximizes F1 score on validation set"""
    print("Optimizing classification threshold...")
    
    # Collect all validation predictions and labels
    all_logits = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Collecting validation predictions"):
            features = batch['features'].to(device)
            binary_labels = batch['binary_label']
            
            # Get audio features if available
            audio_features = batch['audio_features'].to(device) if 'audio_features' in batch else None
            
            # Forward pass
            logits = model(features, audio_features)
            
            # Collect results
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(binary_labels.numpy())
    
    # Convert to numpy arrays
    all_logits = np.array(all_logits).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Try different threshold values
    thresholds = np.linspace(-3, 3, 61)  # Try values from -3 to 3 (logit scale) in 0.1 increments
    f1_scores = []
    precisions = []
    recalls = []
    
    # Calculate F1 score for each threshold
    for threshold in thresholds:
        preds = (all_logits > threshold).astype(int)
        
        # Handle potential edge cases with no positive predictions
        if np.sum(preds) == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = np.sum((preds == 1) & (all_labels == 1)) / np.sum(preds)
            recall = np.sum((preds == 1) & (all_labels == 1)) / np.sum(all_labels)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    
    # Find threshold with highest F1 score
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Convert to probability scale for interpretation
    optimal_prob = 1 / (1 + np.exp(-optimal_threshold))
    
    print(f"Optimal threshold: {optimal_threshold:.4f} (probability: {optimal_prob:.4f})")
    print(f"Validation F1 at optimal threshold: {best_f1:.4f}")
    print(f"Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}")
    
    # Plot threshold vs F1 curve
    plt.figure(figsize=(10, 6))
    
    # F1 score curve
    plt.subplot(2, 1, 1)
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle=':')  # Default threshold
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    plt.subplot(2, 1, 2)
    plt.plot(thresholds, precisions, 'g-', label='Precision')
    plt.plot(thresholds, recalls, 'm-', label='Recall')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    plt.axvline(x=0, color='gray', linestyle=':', label='Default (0.00)')
    plt.xlabel('Threshold (logit scale)')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "threshold_optimization.png"), dpi=300)
    plt.close()
    
    # Save threshold info
    threshold_info = {
        'optimal_threshold': float(optimal_threshold),
        'optimal_probability': float(optimal_prob),
        'threshold_f1': float(best_f1),
        'threshold_precision': float(precisions[best_idx]),
        'threshold_recall': float(recalls[best_idx])
    }
    
    with open(os.path.join(OUTPUT_DIR, "optimal_threshold.json"), 'w') as f:
        json.dump(threshold_info, f, indent=4)
    
    return optimal_threshold

# Custom learning rate scheduler with warm-up
class WarmupScheduler:
    """
    Custom scheduler implementing warm-up followed by ReduceLROnPlateau behavior
    
    During warm-up phase, LR increases linearly from 10% to 100% of target LR.
    After warm-up, behaves like ReduceLROnPlateau, reducing LR when improvements plateau.
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, factor=0.7, patience=7, 
                 min_lr=1e-6, verbose=False):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        # For plateau detection
        self.best_score = -float('inf')
        self.bad_epochs = 0
        self.last_epoch = -1
        
        # For warm-up phase - start at 10% of base LR
        self._set_lr(base_lr * 0.1)
    
    def _set_lr(self, lr):
        """Set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(lr, self.min_lr)  # Ensure LR doesn't go below minimum
    
    def step(self, epoch, val_metric=None):
        """Update learning rate based on epoch and validation metric"""
        self.last_epoch = epoch
        
        # Warm-up phase: linear increase from 10% to 100% of base LR
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            lr = self.base_lr * (0.1 + 0.9 * progress)  # Linear warm-up
            self._set_lr(lr)
            
            if self.verbose:
                print(f"Warm-up LR: {lr:.6f} ({progress*100:.1f}% of warm-up complete)")
            return
        
        # After warm-up: use ReduceLROnPlateau behavior
        if val_metric is None:
            return  # No metric provided, keep current LR
        
        # Check if validation metric improved
        if val_metric > self.best_score + 1e-4:  # Small threshold for numerical stability
            improvement = val_metric - self.best_score
            self.best_score = val_metric
            self.bad_epochs = 0
            
            if self.verbose and epoch >= self.warmup_epochs:
                print(f"Validation metric improved by {improvement:.4f}, resetting patience")
        else:
            self.bad_epochs += 1
            
            if self.verbose:
                print(f"Validation metric did not improve. Patience: {self.bad_epochs}/{self.patience}")
            
            # If patience is exhausted, reduce learning rate
            if self.bad_epochs >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                
                # Reset patience counter
                self.bad_epochs = 0
                
                if self.verbose:
                    print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")

def train():
    # Get dataloaders
    data = get_dataloaders(
        data_dir=DATA_DIR,
        labels_dir=LABELS_DIR,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        include_movement_features=INCLUDE_MOVEMENT_FEATURES,
        include_pose=INCLUDE_POSE,
        include_audio=INCLUDE_AUDIO  # Pass audio flag to data processor
    )
    
    # Extract loaders and datasets
    train_loader = data['train']['loader']
    dev_loader = data['dev']['loader']
    test_loader = data['test']['loader']
    
    train_dataset = data['train']['dataset']
    
    # Get input size and feature names from dataset
    if len(train_dataset) > 0:
        sample_batch = next(iter(train_loader))
        input_size = sample_batch['features'].shape[2]
        feature_names = train_dataset.get_feature_names()
        
        # Get audio feature names if available
        audio_feature_names = train_dataset.get_audio_feature_names() if INCLUDE_AUDIO else []
    else:
        # Default values if dataset is empty
        input_size = 75
        feature_names = [f"feature_{i}" for i in range(input_size)]
        audio_feature_names = []
    
    # Initialize model with new fusion type parameter
    model = HybridDepBinaryClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pose_dropout=POSE_DROPOUT,
        audio_dropout=AUDIO_DROPOUT,
        seq_length=MAX_SEQ_LENGTH,
        feature_names=feature_names,
        audio_feature_names=audio_feature_names,
        include_pose=INCLUDE_POSE,
        include_audio=INCLUDE_AUDIO,
        use_hybrid_pathways=USE_HYBRID_PATHWAYS,
        fusion_type=FUSION_TYPE  # Pass new fusion_type parameter
    ).to(device)
    
    # Calculate class weights based on class distribution (24% depressed, 76% non-depressed)
    pos_weight = torch.tensor([3.16]).to(device)
    
    # Define loss function with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use AdamW optimizer with higher weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Use custom learning rate scheduler with warm-up
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        base_lr=LEARNING_RATE,
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        verbose=True
    )
    
    # For early stopping - initialize to negative value so first epoch always improves
    best_val_f1 = -1
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    # Create history CSV file path
    history_csv_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            features = batch['features'].to(device)
            binary_labels = batch['binary_label'].to(device)
            
            # Get audio features if available
            audio_features = batch['audio_features'].to(device) if 'audio_features' in batch else None
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(features, audio_features)  # Pass audio features to model
            
            # Calculate loss with class weighting
            loss = criterion(logits, binary_labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # More aggressive gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            batch_metrics = model.calculate_metrics(logits, binary_labels)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]
        
        # Average training metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                features = batch['features'].to(device)
                binary_labels = batch['binary_label'].to(device)
                
                # Get audio features if available
                audio_features = batch['audio_features'].to(device) if 'audio_features' in batch else None
                
                # Forward pass
                logits = model(features, audio_features)  # Pass audio features to model
                
                # Calculate loss
                loss = criterion(logits, binary_labels)
                
                # Update metrics
                val_loss += loss.item()
                batch_metrics = model.calculate_metrics(logits, binary_labels)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
        
        # Average validation metrics
        val_loss /= len(dev_loader)
        for key in val_metrics:
            val_metrics[key] /= len(dev_loader)
        
        # Update history
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Update learning rate using our custom scheduler
        scheduler.step(epoch, val_metrics['f1'])
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check if this is the best model so far
        is_best_model = val_metrics['f1'] > best_val_f1 + EARLY_STOPPING_MIN_DELTA
        
        # Save epoch metrics to CSV
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time_seconds': epoch_time,
            'best_model': is_best_model  # Add indicator for best model
        }
        
        # Convert to DataFrame
        epoch_df = pd.DataFrame([epoch_metrics])
        
        # If first epoch, create new file with header, otherwise append without header
        if epoch == 0:
            epoch_df.to_csv(history_csv_path, mode='w', index=False)
        else:
            epoch_df.to_csv(history_csv_path, mode='a', header=False, index=False)
        
        # Check for improvement
        if is_best_model:
            improvement_amount = val_metrics['f1'] - best_val_f1
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            
            # Save metrics as JSON
            metrics_json = {
                'epoch': epoch + 1,
                'train_metrics': {
                    'loss': train_loss,
                    'accuracy': train_metrics['accuracy'],
                    'precision': train_metrics['precision'],
                    'recall': train_metrics['recall'],
                    'f1': train_metrics['f1']
                },
                'validation_metrics': {
                    'loss': val_loss,
                    'accuracy': val_metrics['accuracy'],
                    'precision': val_metrics['precision'],
                    'recall': val_metrics['recall'],
                    'f1': val_metrics['f1']
                },
                'hyperparameters': config
            }
            
            # Save metrics to JSON file
            metrics_path = os.path.join(OUTPUT_DIR, "best_model_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_json, f, indent=4)
            
            print(f"  Model improved by {improvement_amount:.4f}! Saved checkpoint and metrics to {metrics_path}")
        else:
            patience_counter += 1
            print(f"  No significant improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping! No significant improvement for {EARLY_STOPPING_PATIENCE} epochs")
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    
    # Optimize threshold on validation set
    optimal_threshold = optimize_threshold(model, dev_loader)
    
    # Evaluate on test set with optimized threshold
    evaluate_model(model, test_loader, optimal_threshold)
    
    # Analyze feature importance
    analyze_feature_importance(model, test_loader)
    
    return model

# Use the same plotting, evaluation and analysis functions as before
# ... existing evaluation code from other models ...
def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(16, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=300)
    plt.close()

def evaluate_model(model, test_loader, threshold=0.0):
    """Evaluate model on test set and generate reports using specified threshold"""
    model.eval()
    
    print(f"Evaluating model with threshold = {threshold:.4f}")
    
    all_predictions = []
    all_logits = []
    all_labels = []
    all_participant_ids = []
    all_phq_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            features = batch['features'].to(device)
            binary_labels = batch['binary_label']
            participant_ids = batch['participant_id']
            
            # Get audio features if available
            audio_features = batch['audio_features'].to(device) if 'audio_features' in batch else None
            
            # Forward pass
            logits = model(features, audio_features)  # Pass audio features
            probs = torch.sigmoid(logits)
            
            # Collect results
            all_logits.extend(logits.cpu().numpy())
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(binary_labels.numpy())
            all_participant_ids.extend(participant_ids)
            
            # Collect PHQ scores if available
            if 'phq_score' in batch:
                all_phq_scores.extend(batch['phq_score'].numpy())
    
    # Convert to numpy arrays
    all_logits = np.array(all_logits).flatten()
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate metrics using the optimized threshold
    binary_preds = (all_logits > threshold).astype(int)
    
    # For reference, also calculate metrics with default threshold (0.0)
    default_preds = (all_logits > 0.0).astype(int)
    
    # Calculate ROC AUC (threshold-independent)
    auc = roc_auc_score(all_labels, all_predictions)
    
    # Calculate confusion matrices for both thresholds
    cm = confusion_matrix(all_labels, binary_preds)
    cm_default = confusion_matrix(all_labels, default_preds)
    
    # Plot confusion matrix with optimized threshold
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-depressed', 'Depressed'],
                yticklabels=['Non-depressed', 'Depressed'])
    plt.title(f'Confusion Matrix (Threshold = {threshold:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Generate classification reports for both thresholds
    report = classification_report(all_labels, binary_preds, 
                                 target_names=['Non-depressed', 'Depressed'],
                                 output_dict=True)
    
    report_default = classification_report(all_labels, default_preds, 
                                         target_names=['Non-depressed', 'Depressed'],
                                         output_dict=True)
    
    # Save optimized threshold report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
    
    # Compare the two thresholds (optimal vs. default)
    threshold_comparison = {
        'optimal_threshold': {
            'threshold': threshold,
            'accuracy': report['accuracy'],
            'precision': report['Depressed']['precision'],
            'recall': report['Depressed']['recall'],
            'f1': report['Depressed']['f1-score'],
            'confusion_matrix': cm.tolist()
        },
        'default_threshold': {
            'threshold': 0.0,
            'accuracy': report_default['accuracy'],
            'precision': report_default['Depressed']['precision'],
            'recall': report_default['Depressed']['recall'],
            'f1': report_default['Depressed']['f1-score'],
            'confusion_matrix': cm_default.tolist()
        },
        'improvement': {
            'accuracy': report['accuracy'] - report_default['accuracy'],
            'precision': report['Depressed']['precision'] - report_default['Depressed']['precision'],
            'recall': report['Depressed']['recall'] - report_default['Depressed']['recall'],
            'f1': report['Depressed']['f1-score'] - report_default['Depressed']['f1-score']
        }
    }
    
    # Save threshold comparison to JSON
    with open(os.path.join(OUTPUT_DIR, "threshold_comparison.json"), 'w') as f:
        json.dump(threshold_comparison, f, indent=4)
    
    # Print report (for optimized threshold)
    print("\nTest Set Evaluation:")
    print(f"  Using optimized threshold: {threshold:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Precision (Depressed): {report['Depressed']['precision']:.4f}")
    print(f"  Recall (Depressed): {report['Depressed']['recall']:.4f}")
    print(f"  F1 Score (Depressed): {report['Depressed']['f1-score']:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    
    print("\nImprovement over default threshold:")
    print(f"  Accuracy: {threshold_comparison['improvement']['accuracy']:.4f}")
    print(f"  Precision: {threshold_comparison['improvement']['precision']:.4f}")
    print(f"  Recall: {threshold_comparison['improvement']['recall']:.4f}")
    print(f"  F1 Score: {threshold_comparison['improvement']['f1']:.4f}")
    
    # Create test metrics JSON (using optimized threshold)
    test_metrics = {
        'threshold': float(threshold),
        'accuracy': report['accuracy'],
        'precision_depressed': report['Depressed']['precision'],
        'recall_depressed': report['Depressed']['recall'],
        'f1_depressed': report['Depressed']['f1-score'],
        'roc_auc': auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'threshold_comparison': threshold_comparison
    }
    
    # Save test metrics to JSON file
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f"  Test metrics saved to: {metrics_path}")
    
    # Prepare results DataFrame with both logits, probabilities, and both threshold predictions
    results_dict = {
        'Participant_ID': all_participant_ids,
        'True_Label': all_labels,
        'Logits': all_logits,
        'Predicted_Prob': all_predictions,
        'Predicted_Label_Default': default_preds,
        'Predicted_Label_Optimized': binary_preds
    }
    
    # Add PHQ scores if available
    if all_phq_scores:
        results_dict['PHQ8_Score'] = all_phq_scores
        
        # If we have PHQ scores, plot additional analysis
        if len(all_phq_scores) > 0:
            plot_score_analysis(np.array(all_phq_scores), all_labels, binary_preds)
    
    # Save predictions to CSV
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

def plot_score_analysis(phq_scores, true_labels, pred_labels):
    """Plot analysis of predictions by PHQ score (if available)"""
    # Create dataframe
    df = pd.DataFrame({
        'PHQ8_Score': phq_scores,
        'True_Label': true_labels,
        'Predicted_Label': pred_labels,
        'Correct': (true_labels == pred_labels).astype(int)
    })
    
    # Group by PHQ score
    grouped = df.groupby('PHQ8_Score').agg({
        'Correct': 'mean',
        'True_Label': 'count'
    }).reset_index()
    grouped.columns = ['PHQ8_Score', 'Accuracy', 'Count']
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Accuracy by PHQ score
    plt.subplot(1, 2, 1)
    plt.bar(grouped['PHQ8_Score'], grouped['Accuracy'], color='skyblue')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.xlabel('PHQ-8 Score')
    plt.ylabel('Classification Accuracy')
    plt.title('Classification Accuracy by PHQ-8 Score')
    
    # Count by PHQ score
    plt.subplot(1, 2, 2)
    plt.bar(grouped['PHQ8_Score'], grouped['Count'], color='lightgreen')
    plt.xlabel('PHQ-8 Score')
    plt.ylabel('Sample Count')
    plt.title('Sample Distribution by PHQ-8 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phq_score_analysis.png"), dpi=300)
    plt.close()

def analyze_feature_importance(model, test_loader):
    """Analyze and visualize feature importance"""
    # Get a batch of test data
    batch = next(iter(test_loader))
    features = batch['features'].to(device)
    audio_features = batch['audio_features'].to(device) if 'audio_features' in batch else None
    
    print("Analyzing feature importance...")
    print(f"Feature batch shape: {features.shape}")
    print(f"Feature values range: [{features.min().item():.4f}, {features.max().item():.4f}]")
    
    # Get model's attention weights
    attention_weights = model.get_attention_weights()
    print(f"Retrieved {len(attention_weights)} attention weight tensors")
    
    # To debug gradient flow, use a single example first
    single_example = features[0:1]  # Just the first example in batch
    print(f"Using single example of shape {single_example.shape} for initial analysis")
    
    # Calculate global feature importance using integrated gradients
    print("Calculating integrated gradients (this may take a moment)...")
    importance_dict = model.integrated_gradients(
        features[0:1],  # Use single example
        audio_features=audio_features[0:1] if audio_features is not None else None,
        steps=20
    )
    
    # Print importance stats
    importance_values = list(importance_dict.values())
    print(f"Feature importance stats: min={min(importance_values):.6f}, max={max(importance_values):.6f}")
    
    # Save global feature importance to JSON
    global_importance_sorted = {k: float(v) for k, v in 
                              sorted(importance_dict.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True)}
    
    # Add metadata
    global_importance_json = {
        'importance_scores': global_importance_sorted,
        'metadata': {
            'method': 'integrated_gradients',
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'include_pose': model.include_pose,
                'include_audio': model.include_audio,
                'use_hybrid_pathways': model.use_hybrid_pathways
            },
            'top_features': list(global_importance_sorted.keys())[:20],
            'statistics': {
                'min': float(min(importance_values)),
                'max': float(max(importance_values)),
                'mean': float(np.mean(importance_values)),
                'median': float(np.median(importance_values))
            }
        }
    }
    
    # Save to JSON file
    with open(os.path.join(OUTPUT_DIR, "global_feature_importance.json"), 'w') as f:
        json.dump(global_importance_json, f, indent=4)
    
    # Visualize global feature importance
    print("Visualizing feature importance...")
    model.visualize_feature_importance(
        importance_dict=importance_dict,
        top_k=20,
        save_path=os.path.join(OUTPUT_DIR, "global_feature_importance.png"),
        title="Global Feature Importance for Depression Detection (Hybrid Model)",
        highlight_audio=True
    )
    
    # If we have audio features, generate audio-specific visualization
    if audio_features is not None:
        audio_importance = model.get_audio_feature_importance()
        if audio_importance:
            # Sort by importance
            audio_importance = {k: v for k, v in 
                              sorted(audio_importance.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True)}
            
            # Plot top audio features
            plt.figure(figsize=(12, 8))
            plt.barh(list(audio_importance.keys())[:15], 
                     list(audio_importance.values())[:15], 
                     color='purple')
            plt.xlabel('Importance')
            plt.title('Top Audio Features for Depression Detection')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "audio_feature_importance.png"), dpi=300)
            plt.close()
    
    # Get modality importance if available
    modality_importance = model.get_modality_importance()
    if modality_importance:
        # Define consistent colors for modality types
        color_map = {
            'pose': 'green', 
            'facial_expressions': 'blue',
            'audio': 'purple',
            'au_gaze_temporal': 'orange',  # For AU/gaze through xLSTM
            'au_gaze_spatial': 'blue'      # For AU/gaze through CNN
        }
        
        # Get colors for the current modalities
        colors = [color_map.get(modality, 'gray') for modality in modality_importance.keys()]
        
        # Create descriptive labels for the plot
        label_map = {
            'pose': 'Pose (xLSTM)',
            'facial_expressions': 'Facial Expressions (CNN)', 
            'audio': 'Audio (xLSTM)',
            'au_gaze_temporal': 'AU/Gaze Temporal (xLSTM)',
            'au_gaze_spatial': 'AU/Gaze Spatial (CNN)'
        }
        
        # Map keys to more descriptive labels for visualization
        plot_labels = [label_map.get(modality, modality) for modality in modality_importance.keys()]
        
        # Plot modality importance
        plt.figure(figsize=(10, 6))
        plt.bar(plot_labels, modality_importance.values(), color=colors)
        plt.xlabel('Modality')
        plt.ylabel('Relative Importance')
        plt.title('Relative Importance of Different Pathways')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "modality_importance.png"), dpi=300)
        plt.close()
        
        # Create a copy of modality_importance with original keys for JSON serialization
        # Convert NumPy float32 values to native Python floats for JSON serialization
        modality_importance_json = {k: float(v) for k, v in modality_importance.items()}
        
        # Also include descriptive labels in the JSON
        modality_importance_json_with_labels = {
            'importance': modality_importance_json,
            'descriptive_labels': {k: label_map.get(k, k) for k in modality_importance.keys()}
        }
        
        # Save modality importance to file
        with open(os.path.join(OUTPUT_DIR, "modality_importance.json"), 'w') as f:
            json.dump(modality_importance_json_with_labels, f, indent=4)
    
    # Try gradient-based method too for comparison
    print("Calculating gradient-based importance...")
    single_audio_features = audio_features[0:1] if audio_features is not None else None
    grad_importance = model.gradient_feature_importance(
        single_example, 
        audio_features=single_audio_features
    )
    
    model.visualize_feature_importance(
        importance_dict=grad_importance,
        top_k=20,
        save_path=os.path.join(OUTPUT_DIR, "gradient_based_importance.png"),
        title="Gradient-Based Feature Importance (Hybrid Model)"
    )
    
    # Extract and visualize importance of clinically significant AUs
    print("Analyzing clinical AU importance...")
    clinical_importance = model.get_clinical_au_importance()
    
    # Print clinical importance values
    print("Clinical AU Importance:")
    for au, importance in clinical_importance.items():
        print(f"  {au}: {importance:.6f}")
    
    # Plot clinical AUs importance
    plt.figure(figsize=(10, 6))
    bars = plt.barh(list(clinical_importance.keys()), list(clinical_importance.values()), color='orangered')
    plt.xlabel('Importance Score')
    plt.title('Importance of Clinically Significant Action Units (CNN Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "clinical_au_importance.png"), dpi=300)
    plt.close()
    
    # Save clinical importance to CSV
    clinical_df = pd.DataFrame({
        'AU': list(clinical_importance.keys()),
        'Importance': list(clinical_importance.values())
    })
    clinical_df.to_csv(os.path.join(OUTPUT_DIR, "clinical_au_importance.csv"), index=False)
    
    # Calculate instance-specific feature importance for a few examples
    print("Analyzing instance-specific examples...")
    
    depressed_batch = None
    non_depressed_batch = None
    
    # Find examples of each class
    for batch in test_loader:
        labels = batch['binary_label'].numpy().flatten()
        # Check for audio features in the batch
        batch_audio = batch['audio_features'].to(device) if 'audio_features' in batch else None
        
        if 1 in labels and depressed_batch is None:
            # Get first depressed example
            idx = np.where(labels == 1)[0][0]
            depressed_batch = {
                'features': batch['features'][idx:idx+1].to(device),
                'audio_features': batch_audio[idx:idx+1] if batch_audio is not None else None,
                'participant_id': batch['participant_id'][idx],
                'phq_score': batch['phq_score'][idx].item() if 'phq_score' in batch else 'N/A'
            }
        
        if 0 in labels and non_depressed_batch is None:
            # Get first non-depressed example
            idx = np.where(labels == 0)[0][0]
            non_depressed_batch = {
                'features': batch['features'][idx:idx+1].to(device),
                'audio_features': batch_audio[idx:idx+1] if batch_audio is not None else None,
                'participant_id': batch['participant_id'][idx],
                'phq_score': batch['phq_score'][idx].item() if 'phq_score' in batch else 'N/A'
            }
        
        if depressed_batch is not None and non_depressed_batch is not None:
            break
    
    # Analyze depressed example
    if depressed_batch is not None:
        dep_importance = model.instance_feature_importance(
            depressed_batch['features'], 
            depressed_batch['audio_features']
        )
        
        phq_info = f" (PHQ-8: {depressed_batch['phq_score']})" if depressed_batch['phq_score'] != 'N/A' else ""
        
        model.visualize_feature_importance(
            importance_dict=dep_importance['importance'],
            top_k=15,
            save_path=os.path.join(OUTPUT_DIR, f"instance_importance_depressed_{depressed_batch['participant_id']}.png"),
            title=f"Feature Importance for Depressed Participant {depressed_batch['participant_id']}{phq_info}"
        )
    
    # Analyze non-depressed example
    if non_depressed_batch is not None:
        nondep_importance = model.instance_feature_importance(
            non_depressed_batch['features'],
            non_depressed_batch['audio_features']
        )
        
        phq_info = f" (PHQ-8: {non_depressed_batch['phq_score']})" if non_depressed_batch['phq_score'] != 'N/A' else ""
        
        model.visualize_feature_importance(
            importance_dict=nondep_importance['importance'],
            top_k=15,
            save_path=os.path.join(OUTPUT_DIR, f"instance_importance_nondepressed_{non_depressed_batch['participant_id']}.png"),
            title=f"Feature Importance for Non-depressed Participant {non_depressed_batch['participant_id']}{phq_info}"
        )
    
    # Compare feature importance between depressed and non-depressed participants
    if depressed_batch is not None and non_depressed_batch is not None:
        comparison_data = compare_feature_importance(
            dep_importance['importance'],
            nondep_importance['importance'],
            depressed_batch['participant_id'],
            non_depressed_batch['participant_id'],
            save_path=os.path.join(OUTPUT_DIR, "feature_importance_comparison.png"),
            save_json=True,  # Add parameter to save JSON
            output_dir=OUTPUT_DIR  # Pass output directory
        )

def compare_feature_importance(dep_importance, nondep_importance, dep_id, nondep_id, save_path=None, save_json=False, output_dir=None):
    """Compare feature importance between depressed and non-depressed participants"""
    # Get top 10 features for each
    dep_top = dict(sorted(dep_importance.items(), key=lambda x: x[1], reverse=True)[:10])
    nondep_top = dict(sorted(nondep_importance.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Combine unique features
    all_features = list(set(list(dep_top.keys()) + list(nondep_top.keys())))
    
    # Create comparison values
    dep_values = [dep_top.get(feature, 0) for feature in all_features]
    nondep_values = [nondep_top.get(feature, 0) for feature in all_features]
    
    # Sort by average importance
    avg_importance = [(dep_values[i] + nondep_values[i])/2 for i in range(len(all_features))]
    sorted_indices = np.argsort(avg_importance)[::-1]
    
    all_features = [all_features[i] for i in sorted_indices]
    dep_values = [dep_values[i] for i in sorted_indices]
    nondep_values = [nondep_values[i] for i in sorted_indices]
    
    # Create comparison data dictionary
    comparison_data = {
        'features': all_features,
        'depressed': {
            'participant_id': dep_id,
            'importance_values': [float(val) for val in dep_values]
        },
        'non_depressed': {
            'participant_id': nondep_id,
            'importance_values': [float(val) for val in nondep_values]
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'method': 'integrated_gradients_instance'
        }
    }
    
    # Save to JSON if requested
    if save_json and output_dir:
        json_path = os.path.join(output_dir, "feature_importance_comparison.json")
        with open(json_path, 'w') as f:
            json.dump(comparison_data, f, indent=4)
            print(f"Feature importance comparison saved to: {json_path}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(all_features))
    width = 0.35
    
    plt.barh(x + width/2, dep_values, width, label=f'Depressed (ID: {dep_id})', color='tomato')
    plt.barh(x - width/2, nondep_values, width, label=f'Non-depressed (ID: {nondep_id})', color='cornflowerblue')
    
    plt.yticks(x, all_features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Comparison: Depressed vs. Non-depressed (CNN Model)')
    
    # Enhance legend with more descriptive labels
    plt.legend(loc='lower right', title='Participant Status')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_data  # Return the data for potential further use

if __name__ == "__main__":
    train()
