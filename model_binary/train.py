import os
import random  # Add random module import
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime

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
    print(f"Random seed set to {seed} for reproducibility")

# Call set_seed at the beginning
set_seed()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 16
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_SEQ_LENGTH = 250
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Directories
DATA_DIR = "E-DAIC/data_extr"
LABELS_DIR = "E-DAIC/labels"
timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
OUTPUT_DIR = f"model_binary/results/binary_{timestamp}"
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
    'num_epochs': NUM_EPOCHS,
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'device': str(device)
}

# Save configuration
with open(os.path.join(OUTPUT_DIR, 'config.txt'), 'w') as f:
    for key, value in config.items():
        f.write(f"{key}: {value}\n")

def train():
    # Get dataloaders
    data = get_dataloaders(
        data_dir=DATA_DIR,
        labels_dir=LABELS_DIR,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH
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
        feature_names=feature_names
    ).to(device)
    
    # Calculate class weights based on class distribution (24% depressed, 76% non-depressed)
    # Weight for positive class: 76/24 ≈ 3.16
    pos_weight = torch.tensor([3.16]).to(device)
    
    # Define loss function with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
        'val_f1': []
    }
    
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
            
            # Calculate loss with class weighting via BCEWithLogitsLoss
            loss = criterion(logits, binary_labels)
            
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
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Check for improvement
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"  Model improved! Saved checkpoint.")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping!")
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    
    # Evaluate on test set
    evaluate_model(model, test_loader)
    
    # Analyze feature importance
    analyze_feature_importance(model, test_loader)
    
    return model

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

def evaluate_model(model, test_loader):
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
    binary_preds = (all_predictions > 0.5).astype(int)
    
    # Calculate ROC AUC
    auc = roc_auc_score(all_labels, all_predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-depressed', 'Depressed'],
                yticklabels=['Non-depressed', 'Depressed'])
    plt.title('Confusion Matrix')
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
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Precision (Depressed): {report['Depressed']['precision']:.4f}")
    print(f"  Recall (Depressed): {report['Depressed']['recall']:.4f}")
    print(f"  F1 Score (Depressed): {report['Depressed']['f1-score']:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    
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
    
    print("Analyzing feature importance...")
    
    # Get model's attention weights
    model.get_attention_weights()
    
    # Calculate global feature importance using integrated gradients
    importance_dict = model.integrated_gradients(features)
    
    # Visualize global feature importance
    model.visualize_feature_importance(
        importance_dict=importance_dict,
        top_k=20,
        save_path=os.path.join(OUTPUT_DIR, "global_feature_importance.png"),
        title="Global Feature Importance for Depression Detection"
    )
    
    # Extract and visualize importance of clinically significant AUs
    clinical_importance = model.get_clinical_au_importance()
    
    # Plot clinical AUs importance
    plt.figure(figsize=(10, 6))
    bars = plt.barh(list(clinical_importance.keys()), list(clinical_importance.values()), color='orangered')
    plt.xlabel('Importance Score')
    plt.title('Importance of Clinically Significant Action Units')
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
    plt.title('Feature Importance Comparison: Depressed vs. Non-depressed')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train()
