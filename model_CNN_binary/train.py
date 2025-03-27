import os
import random  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, f1_score
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
import json

from model import DepBinaryClassifier
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
HIDDEN_SIZE = 256
NUM_LAYERS = 3  # Number of CNN layers
DROPOUT = 0.3
MAX_SEQ_LENGTH = 750  
LEARNING_RATE = 0.001  # Reduced from 0.01 to 0.001 for stability
WEIGHT_DECAY = 1e-5    # Added weight decay for regularization
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
INCLUDE_MOVEMENT_FEATURES = False  
INCLUDE_POSE = False  
POSE_SCALING_FACTOR = 0.1  # More aggressive scaling (reduced from 0.2 to 0.1)
POSE_REG_STRENGTH = 0.5  # Increased from 0.01 to 0.02 for stronger regularization
LR_PATIENCE = 4  # Epochs to wait before reducing learning rate
LR_FACTOR = 0.5  # Factor to reduce learning rate by
LOSS_TYPE = 'focal'    # Options: 'bce', 'focal'
FOCAL_GAMMA = 2.0      # Focal loss focusing parameter
FOCAL_ALPHA = 0.75     # Focal loss alpha parameter for positive class weight

# Directories
DATA_DIR = "E-DAIC/data_extr"
LABELS_DIR = "E-DAIC/labels"
timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
OUTPUT_DIR = f"model_CNN_binary/results/binary_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration logging
config = {
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'max_seq_length': MAX_SEQ_LENGTH,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'num_epochs': NUM_EPOCHS,
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'lr_patience': LR_PATIENCE,
    'lr_factor': LR_FACTOR,
    'include_movement_features': INCLUDE_MOVEMENT_FEATURES,
    'include_pose': INCLUDE_POSE,
    'pose_scaling_factor': POSE_SCALING_FACTOR,
    'pose_reg_strength': POSE_REG_STRENGTH,
    'model_type': 'CNN',  # Changed from LSTM
    'device': str(device),
    'loss_type': LOSS_TYPE,
    'focal_gamma': FOCAL_GAMMA,
    'focal_alpha': FOCAL_ALPHA
}

# Save configuration
with open(os.path.join(OUTPUT_DIR, 'config.txt'), 'w') as f:
    for key, value in config.items():
        f.write(f"{key}: {value}\n")

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.
    
    This loss helps focus more on hard examples and less on easy ones by adding
    a modulating factor to standard cross entropy.
    
    Args:
        alpha (float): Weight for the positive class (between 0-1)
        gamma (float): Focusing parameter (>= 0). Higher values increase focus on hard examples.
        reduction (str): Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7  # Small epsilon to avoid log(0) issues
    
    def forward(self, inputs, targets):
        """
        Calculate focal loss
        
        Args:
            inputs (torch.Tensor): Raw logits from the model (not sigmoid-activated)
            targets (torch.Tensor): Target labels (0 or 1)
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy loss
        # For numerical stability, we use the built-in binary_cross_entropy_with_logits later
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate the focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        # Apply the weight to BCE loss
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

def train():
    # Get dataloaders
    data = get_dataloaders(
        data_dir=DATA_DIR,
        labels_dir=LABELS_DIR,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        include_movement_features=INCLUDE_MOVEMENT_FEATURES,
        include_pose=INCLUDE_POSE
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
    else:
        # Default values if dataset is empty
        input_size = 75
        feature_names = [f"feature_{i}" for i in range(input_size)]
    
    # Initialize model
    model = DepBinaryClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        seq_length=MAX_SEQ_LENGTH,
        feature_names=feature_names,
        include_pose=INCLUDE_POSE,
        pose_scaling_factor=POSE_SCALING_FACTOR
    ).to(device)
    
    # Choose loss function based on configuration
    if LOSS_TYPE == 'focal':
        # Use Focal Loss with specified parameters
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        print(f"Using Focal Loss with gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")
    else:
        # Default to BCE with Logits Loss with class weighting
        pos_weight = torch.tensor([3.16]).to(device)  # Class weight: 76/24 ≈ 3.16
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCE Loss with positive class weight={pos_weight.item()}")
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler for reducing LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',        # Monitor F1 score (higher is better)
        factor=LR_FACTOR,  # Multiply LR by this factor on plateau
        patience=LR_PATIENCE,
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
        'train_precision': [],
        'train_recall': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'epoch': [],
        'epoch_time': [],
        'learning_rate': []
    }
    
    # Add feature group weights history if using pose features
    if INCLUDE_POSE:
        history['pose_weight'] = []
        history['gaze_weight'] = []
        history['au_weight'] = []
    
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
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(features)  # Get logits (no sigmoid)
            
            # Calculate loss with class weighting
            loss = criterion(logits, binary_labels)
            
            # Add pose feature regularization loss
            if INCLUDE_POSE:
                reg_loss = model.get_regularization_loss(reg_strength=POSE_REG_STRENGTH)
                loss += reg_loss
            
            # Backward pass and optimize
            loss.backward()
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
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                features = batch['features'].to(device)
                binary_labels = batch['binary_label'].to(device)
                
                # Forward pass
                logits = model(features)  # Get logits
                
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
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Add epoch number and time to history
        history['epoch'].append(epoch + 1)
        history['epoch_time'].append(epoch_time)
        
        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time:.2f}s (Loss: {LOSS_TYPE.upper()})")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # If we're using pose features, log their weights
        if INCLUDE_POSE:
            weights = model.get_feature_group_weights().cpu().numpy()
            print(f"  Feature Group Weights: Pose={weights[0]:.4f}, Gaze={weights[1]:.4f}, AU={weights[2]:.4f}")
            
            # Add feature group weights to history
            history['pose_weight'].append(weights[0])
            history['gaze_weight'].append(weights[1])
            history['au_weight'].append(weights[2])
        
        # Check for improvement
        if val_metrics['f1'] > best_val_f1:
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
            
            print(f"  Model improved! Saved checkpoint and metrics to {metrics_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping!")
                break
        
        # Step the learning rate scheduler
        scheduler.step(val_metrics['f1'])  # Uses validation F1 score
    
    # Save training history to CSV
    save_training_history_csv(history, os.path.join(OUTPUT_DIR, "training_history.csv"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    
    # Find optimal threshold on validation set
    optimal_threshold = find_optimal_threshold(model, dev_loader)
    
    # Evaluate on test set with optimal threshold
    evaluate_model(model, test_loader, threshold=optimal_threshold)
    
    # Analyze feature importance
    analyze_feature_importance(model, test_loader)
    
    return model

def save_training_history_csv(history, file_path):
    """Save complete training history to CSV file"""
    # Convert history dictionary to DataFrame
    history_df = pd.DataFrame(history)
    
    # Add a column to indicate best validation epoch
    best_val_f1_idx = history_df['val_f1'].argmax()
    history_df['is_best'] = False
    history_df.loc[best_val_f1_idx, 'is_best'] = True
    
    # Save to CSV
    history_df.to_csv(file_path, index=False)
    print(f"Saved detailed training history to {file_path}")

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=300)
    plt.close()
    
    # Add a new plot for learning rate
    plt.figure(figsize=(10, 4))
    plt.plot(history['learning_rate'], marker='o')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_rate_history.png"), dpi=300)
    plt.close()
    
    # If pose features are used, plot feature group weights
    if 'pose_weight' in history:
        plt.figure(figsize=(10, 4))
        plt.plot(history['pose_weight'], label='Pose', color='lightcoral')
        plt.plot(history['gaze_weight'], label='Gaze', color='lightblue')
        plt.plot(history['au_weight'], label='AU', color='lightgreen')
        plt.title('Feature Group Weights Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Group Weight')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_weights_history.png"), dpi=300)
        plt.close()

def evaluate_model(model, test_loader, threshold=0.5):
    """Evaluate model on test set and generate reports"""
    model.eval()
    
    all_predictions = []
    all_logits = []
    all_labels = []
    all_participant_ids = []
    all_phq_scores = []  # Only include if PHQ scores are available
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            features = batch['features'].to(device)
            binary_labels = batch['binary_label']
            participant_ids = batch['participant_id']
            
            # Forward pass
            logits = model(features)
            probs = torch.sigmoid(logits)  # Convert to probabilities
            
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
    
    # Calculate metrics - use probabilities for threshold-based metrics
    binary_preds = (all_predictions > threshold).astype(int)
    
    # Calculate ROC AUC
    auc = roc_auc_score(all_labels, all_predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    
    # Plot confusion matrix
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
    
    # Generate classification report
    report = classification_report(all_labels, binary_preds, 
                                 target_names=['Non-depressed', 'Depressed'],
                                 output_dict=True)
    
    # Save report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
    
    # Print report
    print("\nTest Set Evaluation:")
    print(f"  Using threshold: {threshold:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Precision (Depressed): {report['Depressed']['precision']:.4f}")
    print(f"  Recall (Depressed): {report['Depressed']['recall']:.4f}")
    print(f"  F1 Score (Depressed): {report['Depressed']['f1-score']:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    
    # Create test metrics JSON
    test_metrics = {
        'threshold': float(threshold),
        'accuracy': report['accuracy'],
        'precision_depressed': report['Depressed']['precision'],
        'recall_depressed': report['Depressed']['recall'],
        'f1_depressed': report['Depressed']['f1-score'],
        'roc_auc': auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    # Save test metrics to JSON file
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f"  Test metrics saved to: {metrics_path}")
    
    # Prepare results DataFrame with both logits and probabilities
    results_dict = {
        'Participant_ID': all_participant_ids,
        'True_Label': all_labels,
        'Logits': all_logits,
        'Predicted_Prob': all_predictions,
        'Predicted_Label': binary_preds
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
    
    # Plot probability distribution for both classes
    plot_probability_distribution(all_predictions, all_labels, threshold)

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
    labels = batch['binary_label'].to(device)
    
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
    importance_dict = model.integrated_gradients(single_example, steps=20)  # Fewer steps for debugging
    
    # Print importance stats
    importance_values = list(importance_dict.values())
    print(f"Feature importance stats: min={min(importance_values):.6f}, max={max(importance_values):.6f}")
    
    # Save feature importance values to JSON
    importance_json_path = os.path.join(OUTPUT_DIR, "global_feature_importance.json")
    with open(importance_json_path, 'w') as f:
        # Sort by importance value in descending order and convert numpy values to native Python types
        sorted_importance = {}
        for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True):
            # Convert any numpy values to native Python types
            if isinstance(v, np.floating):
                sorted_importance[k] = float(v)
            elif isinstance(v, np.integer):
                sorted_importance[k] = int(v)
            else:
                sorted_importance[k] = v
                
        json.dump(sorted_importance, f, indent=4)
    print(f"Feature importance values saved to {importance_json_path}")
    
    # Visualize global feature importance
    print("Visualizing feature importance...")
    model.visualize_feature_importance(
        importance_dict=importance_dict,
        top_k=20,
        save_path=os.path.join(OUTPUT_DIR, "global_feature_importance.png"),
        title="Global Feature Importance for Depression Detection (CNN Model)"
    )
    
    # Try gradient-based method too for comparison
    print("Calculating gradient-based importance...")
    grad_importance = model.gradient_feature_importance(single_example)
    
    # Save gradient-based importance to JSON
    grad_importance_json_path = os.path.join(OUTPUT_DIR, "gradient_based_importance.json")
    with open(grad_importance_json_path, 'w') as f:
        # Sort by importance value in descending order and convert numpy values to native Python types
        sorted_grad_importance = {}
        for k, v in sorted(grad_importance.items(), key=lambda item: item[1], reverse=True):
            if isinstance(v, np.floating):
                sorted_grad_importance[k] = float(v)
            elif isinstance(v, np.integer):
                sorted_grad_importance[k] = int(v)
            else:
                sorted_grad_importance[k] = v
                
        json.dump(sorted_grad_importance, f, indent=4)
    
    model.visualize_feature_importance(
        importance_dict=grad_importance,
        top_k=20,
        save_path=os.path.join(OUTPUT_DIR, "gradient_based_importance.png"),
        title="Gradient-Based Feature Importance (CNN Model)"
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
        'Importance': [float(v) if isinstance(v, np.floating) else v for v in clinical_importance.values()]
    })
    clinical_df.to_csv(os.path.join(OUTPUT_DIR, "clinical_au_importance.csv"), index=False)
    
    # Calculate instance-specific feature importance for a few examples
    print("Analyzing instance-specific examples...")
    
    depressed_batch = None
    non_depressed_batch = None
    
    # Find examples of each class
    for batch in test_loader:
        labels = batch['binary_label'].numpy().flatten()
        if 1 in labels and depressed_batch is None:
            # Get first depressed example
            idx = np.where(labels == 1)[0][0]
            depressed_batch = {
                'features': batch['features'][idx:idx+1].to(device),
                'participant_id': batch['participant_id'][idx],
                'phq_score': batch['phq_score'][idx].item() if 'phq_score' in batch else 'N/A'
            }
        
        if 0 in labels and non_depressed_batch is None:
            # Get first non-depressed example
            idx = np.where(labels == 0)[0][0]
            non_depressed_batch = {
                'features': batch['features'][idx:idx+1].to(device),
                'participant_id': batch['participant_id'][idx],
                'phq_score': batch['phq_score'][idx].item() if 'phq_score' in batch else 'N/A'
            }
        
        if depressed_batch is not None and non_depressed_batch is not None:
            break
    
    # Analyze depressed example
    if depressed_batch is not None:
        dep_importance = model.instance_feature_importance(depressed_batch['features'])
        
        phq_info = f" (PHQ-8: {depressed_batch['phq_score']})" if depressed_batch['phq_score'] != 'N/A' else ""
        
        # Save instance-specific importance to JSON
        dep_json_path = os.path.join(OUTPUT_DIR, f"instance_importance_depressed_{depressed_batch['participant_id']}.json")
        with open(dep_json_path, 'w') as f:
            # Convert numpy values to native Python types
            sorted_dep_importance = {}
            for k, v in sorted(dep_importance['importance'].items(), key=lambda item: item[1], reverse=True):
                if isinstance(v, np.floating):
                    sorted_dep_importance[k] = float(v)
                elif isinstance(v, np.integer):
                    sorted_dep_importance[k] = int(v)
                else:
                    sorted_dep_importance[k] = v
            
            # Create JSON object with properly converted values
            json_data = {
                'participant_id': depressed_batch['participant_id'],
                'phq_score': float(depressed_batch['phq_score']) if isinstance(depressed_batch['phq_score'], np.floating) else depressed_batch['phq_score'],
                'prediction': float(dep_importance['prediction']) if isinstance(dep_importance['prediction'], np.floating) else dep_importance['prediction'],
                'predicted_class': int(dep_importance['predicted_class']) if isinstance(dep_importance['predicted_class'], np.integer) else dep_importance['predicted_class'],
                'feature_importance': sorted_dep_importance
            }
            
            json.dump(json_data, f, indent=4)
        
        model.visualize_feature_importance(
            importance_dict=dep_importance['importance'],
            top_k=15,
            save_path=os.path.join(OUTPUT_DIR, f"instance_importance_depressed_{depressed_batch['participant_id']}.png"),
            title=f"Feature Importance for Depressed Participant {depressed_batch['participant_id']}{phq_info}"
        )
    
    # Analyze non-depressed example
    if non_depressed_batch is not None:
        nondep_importance = model.instance_feature_importance(non_depressed_batch['features'])
        
        phq_info = f" (PHQ-8: {non_depressed_batch['phq_score']})" if non_depressed_batch['phq_score'] != 'N/A' else ""
        
        # Save instance-specific importance to JSON
        nondep_json_path = os.path.join(OUTPUT_DIR, f"instance_importance_nondepressed_{non_depressed_batch['participant_id']}.json")
        with open(nondep_json_path, 'w') as f:
            # Convert numpy values to native Python types
            sorted_nondep_importance = {}
            for k, v in sorted(nondep_importance['importance'].items(), key=lambda item: item[1], reverse=True):
                if isinstance(v, np.floating):
                    sorted_nondep_importance[k] = float(v)
                elif isinstance(v, np.integer):
                    sorted_nondep_importance[k] = int(v)
                else:
                    sorted_nondep_importance[k] = v
            
            # Create JSON object with properly converted values
            json_data = {
                'participant_id': non_depressed_batch['participant_id'],
                'phq_score': float(non_depressed_batch['phq_score']) if isinstance(non_depressed_batch['phq_score'], np.floating) else non_depressed_batch['phq_score'],
                'prediction': float(nondep_importance['prediction']) if isinstance(nondep_importance['prediction'], np.floating) else nondep_importance['prediction'],
                'predicted_class': int(nondep_importance['predicted_class']) if isinstance(nondep_importance['predicted_class'], np.integer) else nondep_importance['predicted_class'],
                'feature_importance': sorted_nondep_importance
            }
            
            json.dump(json_data, f, indent=4)
        
        model.visualize_feature_importance(
            importance_dict=nondep_importance['importance'],
            top_k=15,
            save_path=os.path.join(OUTPUT_DIR, f"instance_importance_nondepressed_{non_depressed_batch['participant_id']}.png"),
            title=f"Feature Importance for Non-depressed Participant {non_depressed_batch['participant_id']}{phq_info}"
        )
    
    # Compare feature importance between depressed and non-depressed participants
    if depressed_batch is not None and non_depressed_batch is not None:
        compare_feature_importance(
            dep_importance['importance'],
            nondep_importance['importance'],
            depressed_batch['participant_id'],
            non_depressed_batch['participant_id'],
            save_path=os.path.join(OUTPUT_DIR, "feature_importance_comparison.png")
        )
    
    # If we're using pose features, visualize their weight contributions
    if model.include_pose:
        print("Visualizing feature group weights...")
        model.visualize_feature_group_weights(
            save_path=os.path.join(OUTPUT_DIR, "feature_group_weights.png")
        )
    
    # Visualize temporal attention for a few examples
    print("Visualizing temporal attention patterns...")
    
    # Find one example of each class for attention visualization
    depressed_idx = None
    non_depressed_idx = None
    
    for i, label in enumerate(labels):
        if label == 1 and depressed_idx is None:
            depressed_idx = i
        elif label == 0 and non_depressed_idx is None:
            non_depressed_idx = i
            
        if depressed_idx is not None and non_depressed_idx is not None:
            break
    
    # Visualize temporal attention for depressed example
    if depressed_idx is not None:
        print(f"Generating temporal attention visualization for depressed example...")
        dep_features = features[depressed_idx:depressed_idx+1]
        model.visualize_temporal_attention(
            dep_features,
            save_path=os.path.join(OUTPUT_DIR, "temporal_attention_depressed.png")
        )
    
    # Visualize temporal attention for non-depressed example
    if non_depressed_idx is not None:
        print(f"Generating temporal attention visualization for non-depressed example...")
        nondep_features = features[non_depressed_idx:non_depressed_idx+1]
        model.visualize_temporal_attention(
            nondep_features, 
            save_path=os.path.join(OUTPUT_DIR, "temporal_attention_non_depressed.png")
        )
    
    # Add temporal attention visualization to instance-specific analysis
    if depressed_batch is not None:
        # In addition to existing feature importance analysis
        print("Generating temporal attention for depressed instance...")
        model.visualize_temporal_attention(
            depressed_batch['features'],
            save_path=os.path.join(OUTPUT_DIR, f"temporal_attention_depressed_{depressed_batch['participant_id']}.png")
        )
    
    if non_depressed_batch is not None:
        # In addition to existing feature importance analysis
        print("Generating temporal attention for non-depressed instance...")
        model.visualize_temporal_attention(
            non_depressed_batch['features'],
            save_path=os.path.join(OUTPUT_DIR, f"temporal_attention_non_depressed_{non_depressed_batch['participant_id']}.png")
        )

def compare_feature_importance(dep_importance, nondep_importance, dep_id, nondep_id, save_path=None):
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

def find_optimal_threshold(model, val_loader):
    """
    Find the optimal classification threshold that maximizes F1 score on validation set
    
    Args:
        model (nn.Module): Trained model
        val_loader (DataLoader): Validation data loader
        
    Returns:
        float: Optimal threshold for maximizing F1 score
    """
    model.eval()
    
    # Collect all predictions and labels
    all_logits = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Finding optimal threshold"):
            features = batch['features'].to(device)
            binary_labels = batch['binary_label']
            
            # Forward pass
            logits = model(features)
            probs = torch.sigmoid(logits)
            
            # Collect results
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(binary_labels.numpy())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Find the best threshold using precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for i in range(len(thresholds)):
        if precisions[i] + recalls[i] > 0:  # Avoid division by zero
            f1 = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])
            f1_scores.append(f1)
        else:
            f1_scores.append(0)
    
    # Find threshold that maximizes F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Get metrics at this threshold
    binary_preds = (all_probs >= best_threshold).astype(int)
    report = classification_report(all_labels, binary_preds, 
                                  target_names=['Non-depressed', 'Depressed'],
                                  output_dict=True)
    
    print(f"\nOptimal Threshold Analysis:")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  Validation F1 (Depressed): {best_f1:.4f}")
    print(f"  Validation Precision (Depressed): {report['Depressed']['precision']:.4f}")
    print(f"  Validation Recall (Depressed): {report['Depressed']['recall']:.4f}")
    
    # Plot precision-recall curve with optimal threshold
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', label='Precision-Recall curve')
    plt.plot(recalls[best_idx], precisions[best_idx], 'ro', 
             label=f'Optimal threshold = {best_threshold:.4f}, F1 = {best_f1:.4f}')
    
    # Add default threshold of 0.5 for comparison
    default_idx = np.abs(thresholds - 0.5).argmin()
    default_f1 = f1_scores[default_idx]
    plt.plot(recalls[default_idx], precisions[default_idx], 'go',
             label=f'Default threshold = 0.5, F1 = {default_f1:.4f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Optimal Threshold')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "threshold_optimization.png"), dpi=300)
    plt.close()
    
    # Save threshold and metrics to file
    threshold_data = {
        'optimal_threshold': float(best_threshold),
        'default_threshold': 0.5,
        'optimal_f1': float(best_f1),
        'default_f1': float(default_f1),
        'improvement': float(best_f1 - default_f1),
        'precision_at_optimal': float(report['Depressed']['precision']),
        'recall_at_optimal': float(report['Depressed']['recall']),
        'threshold_values': [float(t) for t in thresholds.tolist()],
        'precision_values': [float(p) for p in precisions.tolist()],
        'recall_values': [float(r) for r in recalls.tolist()],
        'f1_values': [float(f) for f in f1_scores]
    }
    
    with open(os.path.join(OUTPUT_DIR, "threshold_optimization.json"), 'w') as f:
        json.dump(threshold_data, f, indent=4)
    
    return best_threshold

def plot_probability_distribution(predictions, labels, threshold):
    """Plot the distribution of predicted probabilities for each class"""
    plt.figure(figsize=(10, 6))
    
    # Get predictions for each class
    depressed_probs = predictions[labels == 1]
    non_depressed_probs = predictions[labels == 0]
    
    # Create the plot
    plt.hist(non_depressed_probs, bins=20, alpha=0.5, label='Non-depressed', color='cornflowerblue')
    plt.hist(depressed_probs, bins=20, alpha=0.5, label='Depressed', color='lightcoral')
    
    # Add threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', 
                label=f'Optimal Threshold = {threshold:.4f}')
    plt.axvline(x=0.5, color='black', linestyle=':', 
                label='Default Threshold = 0.5')
    
    plt.xlabel('Predicted Probability of Depression')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Probabilities by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "probability_distribution.png"), dpi=300)
    plt.close()

# Helper function for safely converting NumPy types to Python native types for JSON serialization
def numpy_to_python_type(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [numpy_to_python_type(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: numpy_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_type(x) for x in obj]
    else:
        return obj

if __name__ == "__main__":
    train()
