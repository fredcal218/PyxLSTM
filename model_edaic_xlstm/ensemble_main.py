import os
import torch
import numpy as np
import random
import json
import time
from datetime import datetime

from model import EDAIC_LSTM
from dataset import EDAICDataset
from ensemble import DepressionEnsemble
from ensemble_trainer import EnsembleTrainer
from trainer import validate_epoch

def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    main_start = time.time()
    print("\n--- Initializing Training Process ---")
    
    # Hard-coded parameters
    base_dir = 'E-DAIC'  
    save_dir = os.path.join('model_edaic_xlstm', 'ensemble_checkpoints')
    epochs = 50
    patience = 5  
    
    # Reduce batch size for ensemble training to prevent OOM
    batch_size = 16
    
    learning_rate = 0.0001
    hidden_size = 256  
    num_layers = 2
    dropout = 0.5
    
    # Use smaller sequence length for ensemble to avoid OOM
    ensemble_seq_length = 900  
    stride = 450
    frame_step = 10
    seed = 46
    
    # Enable gradient accumulation steps to simulate larger batch size
    gradient_accumulation_steps = 4  # Simulate batch size of 32 (8*4)
    
    # Ensure checkpoints directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save hyperparameters to JSON file
    hyperparams = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Depression Ensemble (Binary + Low/High Regressors)",
        "dataset": "E-DAIC",
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "patience": patience,
            "seed": seed,
            "optimizer": "Adam (separate for each model)"
        },
        "model_architecture": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "ensemble_components": [
                "binary_classifier (PHQ-8 ≥ 10)",
                "low_regressor (PHQ-8: 0-9)",
                "high_regressor (PHQ-8: 10-24)"
            ]
        },
        "data_processing": {
            "seq_length": ensemble_seq_length,
            "stride": stride,
            "frame_step": frame_step,
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
    print("Setting random seed for reproducibility...")
    set_seed(seed)
    
    # Create datasets using the predefined splits with reduced sequence length
    print("\nLoading datasets...")
    dataset_start = time.time()
    
    print("Loading training dataset...")
    train_dataset = EDAICDataset(
        base_dir=base_dir,
        split='train',
        seq_length=ensemble_seq_length,
        stride=stride,
        frame_step=frame_step
    )
    
    print("Loading validation dataset...")
    val_dataset = EDAICDataset(
        base_dir=base_dir,
        split='dev',
        seq_length=ensemble_seq_length,
        stride=stride,
        frame_step=frame_step
    )
    
    print("Loading test dataset...")
    test_dataset = EDAICDataset(
        base_dir=base_dir,
        split='test',
        seq_length=ensemble_seq_length,
        stride=stride,
        frame_step=frame_step
    )
    
    print(f"✓ Datasets loaded in {time.time() - dataset_start:.2f}s")
    
    # Check if we have any data
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty. Check dataset paths and structure.")
        return
    
    print(f"Dataset sizes: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Get input size from the data
    sample_data = train_dataset[0][0]
    input_size = sample_data.shape[1]
    
    print(f"Input feature size: {input_size}")
    
    # Create ensemble model with memory optimizations
    print("\nInitializing ensemble model...")
    model_start = time.time()
    model = DepressionEnsemble(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=ensemble_seq_length
    )
    print(f"✓ Model initialized in {time.time() - model_start:.2f}s")
    
    # Create ensemble trainer with gradient accumulation 
    print("\nSetting up trainer...")
    trainer_start = time.time()
    trainer = EnsembleTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_dir,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    print(f"✓ Trainer initialized in {time.time() - trainer_start:.2f}s")
    
    # Train ensemble model
    print("\n--- Starting Ensemble Training ---")
    train_start = time.time()
    history = trainer.train(epochs=epochs, patience=patience)
    train_time = time.time() - train_start
    print(f"\n✓ Training completed in {train_time:.2f}s ({train_time/60:.2f} minutes)")
    
    print("\nEvaluating ensemble model on test set...")
    # Load best model before evaluation
    model_path = os.path.join(save_dir, 'best_ensemble.pt')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.binary_classifier.load_state_dict(checkpoint['binary_classifier'])
        model.low_regressor.load_state_dict(checkpoint['low_regressor'])
        model.high_regressor.load_state_dict(checkpoint['high_regressor'])
        model.confidence_threshold = checkpoint['confidence_threshold']
        print(f"Loaded best ensemble model from {model_path}")
    else:
        print("Warning: No saved model found, using last model state")
    
    # Create test loader with smaller batch for evaluation to prevent OOM
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # Evaluate on test set with memory management strategies
    device = next(model.parameters()).device
    metrics = trainer.evaluate_with_memory_management(test_loader)
    
    test_loss, test_mae, test_rmse, test_r2, test_precision, test_recall, test_f1 = metrics
    
    print("\nTest Set Evaluation:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²: {test_r2:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # Save test results
    test_results = {
        "loss": float(test_loss),
        "mae": float(test_mae),
        "rmse": float(test_rmse),
        "r2": float(test_r2),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1)
    }
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to {os.path.join(save_dir, 'test_results.json')}")
    print(f"\n✓ Total runtime: {(time.time() - main_start)/60:.2f} minutes")
    print("Ensemble training and evaluation completed!")

if __name__ == '__main__':
    print("\nStarting ensemble approach for depression severity prediction...\n")
    main()
