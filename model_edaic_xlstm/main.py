import os
import torch
import numpy as np
import random
import torch.nn as nn
import json
from datetime import datetime

from model import EDAIC_LSTM
from dataset import EDAICDataset
from trainer import Trainer, validate_epoch

def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Hard-coded parameters
    base_dir = 'E-DAIC'  # Base directory for E-DAIC dataset
    save_dir = os.path.join('model_edaic_xlstm', 'checkpoints') # Directory to save model checkpoints
    epochs = 300
    patience = 30  # Reduced from 50 to work better with LR scheduler
    batch_size = 32
    learning_rate = 0.0005
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    seq_length = 9000
    stride = 4500
    seed = 42
    
    # Ensure checkpoints directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save hyperparameters to JSON file
    hyperparams = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "EDAIC_LSTM with xLSTM",
        "dataset": "E-DAIC",
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "seed": seed,
            "optimizer": "Adam",
            "loss_function": "MSE",
            "scheduler": "ReduceLROnPlateau (factor=0.5, patience=15, min_lr=1e-6)"
        },
        "model_architecture": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout
        },
        "data_processing": {
            "seq_length": seq_length,
            "stride": stride,
            "features": "pose, gaze, AU intensities (AU confidences excluded)"
        },
        "hardware": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }
    }
    
    # Save hyperparameters to JSON file
    hyperparams_path = os.path.join(save_dir, 'hyperparameters.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print(f"Hyperparameters saved to {hyperparams_path}")
    
    # Set random seed
    set_seed(seed)
    
    # Create datasets using the predefined splits
    train_dataset = EDAICDataset(
        base_dir=base_dir,
        split='train',
        seq_length=seq_length,
        stride=stride
    )
    
    val_dataset = EDAICDataset(
        base_dir=base_dir,
        split='dev',
        seq_length=seq_length,
        stride=stride
    )
    
    # Add test dataset for final evaluation
    test_dataset = EDAICDataset(
        base_dir=base_dir,
        split='test',
        seq_length=seq_length,
        stride=stride
    )
    
    # Check if we have any data
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty. Check dataset paths and structure.")
        return
    
    print(f"Dataset sizes: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Get input size from the data (first sample's feature dimension)
    sample_data = train_dataset[0][0]
    input_size = sample_data.shape[1]  # Get feature dimension
    
    print(f"Input feature size: {input_size}")
    
    # Create model
    model = EDAIC_LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # or appropriate loss for your regression task
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    # Train model with early stopping and LR scheduling
    history = trainer.train(epochs=epochs, patience=patience, seed=seed)
    
    # Load the best model before the final evaluation
    trainer.load_best_model()
    
    print("\nEvaluating best model on test set:")
    # Create DataLoader for test set evaluation
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    test_loss, test_binary_acc, test_overall_acc, (test_level_acc, test_level_accs), test_rmse, test_mae, test_r2 = test_metrics
    
    # Level names for clearer display
    level_names = ['None/Minimal', 'Mild', 'Moderate', 'Mod. Severe', 'Severe']
    
    print(f"Test set evaluation:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Binary Accuracy: {test_binary_acc:.4f} (depression threshold: 10)")
    print(f"  Overall Accuracy: {test_overall_acc:.4f} (tolerance: ±1)")
    print(f"  PHQ Level Accuracy: {test_level_acc:.4f}")
    print("  Per-level Accuracies:")
    for i, (name, acc) in enumerate(zip(level_names, test_level_accs)):
        print(f"    - {name}: {acc:.4f}")
    print(f"  RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # For comparison, also evaluate on validation set
    # Create a DataLoader for validation set instead of using dataset directly
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    print("\nValidation set results for comparison:")
    val_metrics = validate_epoch(model, val_loader, criterion, device)
    val_loss, val_binary_acc, val_overall_acc, (val_level_acc, val_level_accs), val_rmse, val_mae, val_r2 = val_metrics
    
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Binary Accuracy: {val_binary_acc:.4f} (depression threshold: 10)")
    print(f"  Overall Accuracy: {val_overall_acc:.4f} (tolerance: ±1)")
    print(f"  PHQ Level Accuracy: {val_level_acc:.4f}")
    print("  Per-level Accuracies:")
    for i, (name, acc) in enumerate(zip(level_names, val_level_accs)):
        print(f"    - {name}: {acc:.4f}")
    print(f"  RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    # Save final results with all metrics
    results = {
        "test_metrics": {
            "loss": float(test_loss),
            "binary_accuracy": float(test_binary_acc),
            "overall_accuracy": float(test_overall_acc),
            "level_accuracy": float(test_level_acc),
            "level_accuracies": {
                level_names[i]: float(acc) for i, acc in enumerate(test_level_accs)
            },
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2)
        },
        "val_metrics": {
            "loss": float(val_loss),
            "binary_accuracy": float(val_binary_acc),
            "overall_accuracy": float(val_overall_acc),
            "level_accuracy": float(val_level_acc),
            "level_accuracies": {
                level_names[i]: float(acc) for i, acc in enumerate(val_level_accs)
            },
            "rmse": float(val_rmse),
            "mae": float(val_mae),
            "r2": float(val_r2)
        }
    }
    
    # Add hyperparameters to the results
    results["hyperparameters"] = hyperparams
    
    # Save evaluation results
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {os.path.join(save_dir, 'evaluation_results.json')}")
    print("Training and evaluation completed!")

# Auto-run when file is executed
if __name__ == '__main__':
    print("\nDataset check passed. Starting training...\n")
    main()