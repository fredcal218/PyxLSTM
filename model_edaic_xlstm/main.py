import os
import torch
import numpy as np
import random
import torch.nn as nn

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
    # Get final metrics on the test set with the best model
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    test_loss, test_acc, test_rmse, test_mae, test_r2 = test_metrics
    
    print(f"Test set evaluation:")
    print(f"  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # For comparison, also evaluate on validation set
    print("\nValidation set results for comparison:")
    val_metrics = validate_epoch(model, val_dataset, criterion, device)
    val_loss, val_acc, val_rmse, val_mae, val_r2 = val_metrics
    print(f"  Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    # Save final results
    results = {
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2)
        },
        "val_metrics": {
            "loss": float(val_loss),
            "accuracy": float(val_acc),
            "rmse": float(val_rmse),
            "mae": float(val_mae),
            "r2": float(val_r2)
        }
    }
    
    # Save evaluation results
    import json
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {os.path.join(save_dir, 'evaluation_results.json')}")
    print("Training and evaluation completed!")

# Auto-run when file is executed
if __name__ == '__main__':
    print("\nDataset check passed. Starting training...\n")
    main()