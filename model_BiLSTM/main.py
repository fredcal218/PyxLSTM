import os
import torch
import numpy as np
import random
import torch.nn as nn
import json
from datetime import datetime
import sys

# Add parent directory to path for importing the dataset module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_edaic_xlstm.dataset import EDAICDataset

from bilstm_model import BiLSTM_Attention
from cv_trainer import CrossValidationTrainer

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
    base_dir = 'E-DAIC'
    save_dir = os.path.join('model_BiLSTM', 'checkpoints_cv')
    batch_size = 32
    epochs = 20
    seq_length = 4500
    stride = 2250
    frame_step = 5
    seed = 42
    
    # Ensure checkpoints directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set random seed
    set_seed(seed)
    
    # Create datasets for train, validation, and test
    train_dataset = EDAICDataset(
        base_dir=base_dir,
        split='train',
        seq_length=seq_length,
        stride=stride,
        frame_step=frame_step
    )
    
    val_dataset = EDAICDataset(
        base_dir=base_dir,
        split='dev',
        seq_length=seq_length,
        stride=stride,
        frame_step=frame_step
    )
    
    test_dataset = EDAICDataset(
        base_dir=base_dir,
        split='test',
        seq_length=seq_length,
        stride=stride,
        frame_step=frame_step
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")
    print(f"  Test: {len(test_dataset)} sequences")
    
    # Get input size from the data
    sample_data = train_dataset[0][0]
    input_size = sample_data.shape[1]
    
    print(f"Input feature size: {input_size}")
    
    # Training parameters
    params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'input_size': input_size,
        'seq_length': seq_length,
        'save_dir': save_dir
    }
    
    # Save hyperparameters to JSON file
    hyperparams_metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Bidirectional LSTM with Attention",
        "dataset": "E-DAIC",
        "training": {
            "epochs_per_fold": epochs,
            "batch_size": batch_size,
            "cross_validation": {
                "outer_folds": 5,
                "inner_folds": 5
            },
            "seed": seed,
            "optimizer": "Adam",
            "loss_function": "MSE"
        },
        "model_architecture": {
            "bidirectional": True,
            "attention": True,
            "hyperparameter_search": {
                "hidden_size": [32, 64, 128],
                "num_layers": [2, 3],
                "dropout": [0.2, 0.3, 0.4],
                "learning_rate": [0.01, 0.1]
            }
        },
        "data_processing": {
            "seq_length": seq_length,
            "stride": stride,
            "frame_step": frame_step,
            "features": "31 features (6 head pose, 8 eye gaze, 17 AU intensities)",
            "confidence_threshold": 0.90
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
        json.dump(hyperparams_metadata, f, indent=2)
    
    print(f"Hyperparameters saved to {hyperparams_path}")
    
    # Add command to specify start fold (default is 1)
    start_fold = 1  # Start from fold 2 since fold 1 is already trained
    end_fold = 5    # Process up to fold 5 (can be adjusted as needed)
    
    print(f"Training folds {start_fold} through {end_fold}")
    
    # Create cross-validation trainer
    cv_trainer = CrossValidationTrainer(
        model_class=BiLSTM_Attention,
        train_dataset=train_dataset,
        val_dataset=val_dataset, 
        test_dataset=test_dataset,
        params=params
    )
    
    # Define parameter grid with more stable learning rates
    param_grid = {
        'hidden_size': [32, 64],
        'num_layers': [2, 3],
        'dropout': [0.2, 0.3],
        'learning_rate': [0.1, 0.01]  
    }
    
    # Perform cross-validation, specifying start_fold and end_fold
    print(f"\nStarting nested cross-validation from fold {start_fold} to {end_fold}...")
    results = cv_trainer.perform_cross_validation(
        param_grid=param_grid,
        n_folds=5,
        inner_folds=5,
        start_fold=start_fold,
        end_fold=end_fold
    )
    
    # Print final results
    print("\nCross-validation completed!")
    print("\nBest parameters per fold:")
    for fold, params in enumerate(results['best_params_per_fold']):
        print(f"  Fold {fold+1}: {params}")
    
    print("\nTest metrics (ensemble model):")
    print(f"  RMSE: {results['test_metrics']['rmse']:.4f}")
    print(f"  MAE: {results['test_metrics']['mae']:.4f}")
    print(f"  R²: {results['test_metrics']['r2']:.4f}")
    
    print(f"\nResults saved to {os.path.join(save_dir, 'cv_results.json')}")
    
# Auto-run when file is executed
if __name__ == '__main__':
    print("\nStarting BiLSTM cross-validation for depression severity prediction...\n")
    main()
