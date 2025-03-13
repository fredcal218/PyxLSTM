"""
Training script for depression severity prediction.

This script provides functionality to train and evaluate the depression
predictor model using the E-DAIC dataset.
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from depression_prediction.data_loader import get_data_loaders
from depression_prediction.models import create_model
from depression_prediction.utils import set_seed, get_feature_dimension

# Import for mixed precision training
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Mixed precision training not available")


def train_epoch(model, data_loader, criterion, optimizer, device, scaler=None, gradient_clip=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    nan_encountered = False
    
    # Add progress bar for batches
    progress_bar = tqdm(data_loader, desc="Training batches", leave=False)
    
    for features, labels in progress_bar:
        features = features.to(device)
        labels = labels.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Use mixed precision if available
        if scaler is not None:
            with autocast('cuda'):
                outputs, _ = model(features)
                loss = criterion(outputs, labels)
                
            # Check for NaN loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                nan_encountered = True
                progress_bar.write("NaN/Inf loss detected, skipping batch")
                continue
                
            # Scale loss and calculate gradients
            scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            
            # Check for NaN loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                nan_encountered = True
                progress_bar.write("NaN/Inf loss detected, skipping batch")
                continue
                
            # Calculate gradients
            loss.backward()
            
            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
            # Update weights
            optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss
        
        # Free memory
        torch.cuda.empty_cache()
        
        # Update progress bar description with current loss
        progress_bar.set_postfix(loss=f"{current_loss:.4f}")
    
    # Return average loss and NaN flag
    return total_loss / len(data_loader), nan_encountered


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Add progress bar for evaluation
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for features, labels in progress_bar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    
    return total_loss / len(data_loader), mae, rmse, r2


def train(args):
    """Train and evaluate the model."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Set up device - Force CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Optional: Set this to optimize for your specific GPU
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: No GPU detected. Running on CPU will be significantly slower!")
    
    # Load data
    print("\n=== Loading data ===")
    start_time = time.time()
    train_loader, dev_loader, test_loader = get_data_loaders(
        args.data_dir, args.batch_size, args.max_seq_length, args.stride
    )
    print(f"Data loading complete in {time.time() - start_time:.2f} seconds!")
    
    # Get input feature dimension
    print("\n=== Getting feature dimensions ===")
    input_size = get_feature_dimension(args.data_dir)
    print(f"Feature dimension: {input_size}")
    
    # Create model
    print("\n=== Creating model ===")
    start_time = time.time()
    model_config = {
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_blocks': args.num_blocks,
        'dropout': args.dropout,
        'lstm_type': args.lstm_type
    }
    
    model = create_model(model_config)
    print(f"Moving model to {device}...")
    model = model.to(device)
    print(f"Model creation complete in {time.time() - start_time:.2f} seconds!")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== Model Summary ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Set up loss function and optimizer
    print("\n=== Setting up training components ===")
    criterion = nn.MSELoss()
    print(f"Loss function: MSE")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Optimizer: Adam with learning rate {args.learning_rate}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    print("Scheduler: ReduceLROnPlateau with factor 0.5 and patience 3")
    
    # Set up gradient clipping
    gradient_clip = getattr(args, 'gradient_clip', None)
    if gradient_clip is not None:
        print(f"Gradient clipping: {gradient_clip}")
    
    # Set up mixed precision training
    scaler = None
    if getattr(args, 'mixed_precision', False) and AMP_AVAILABLE and device.type == 'cuda':
        print("Using mixed precision training")
        scaler = GradScaler()
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    print(f"TensorBoard logs directory: {os.path.join(args.output_dir, 'logs')}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Training loop
    print("\n=== Starting Training ===")
    best_dev_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    consecutive_nan_epochs = 0
    
    # Create progress bar for epochs
    epochs_progress = tqdm(range(args.num_epochs), desc="Training epochs", unit="epoch")
    
    for epoch in epochs_progress:
        start_time = time.time()
        
        # Train
        train_loss, nan_encountered = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            scaler=scaler, gradient_clip=gradient_clip
        )
        
        # Check for NaN in training loss
        if nan_encountered or np.isnan(train_loss) or np.isinf(train_loss):
            consecutive_nan_epochs += 1
            epochs_progress.write(f"Warning: NaN/Inf detected in epoch {epoch+1}. ({consecutive_nan_epochs} consecutive)")
            
            # If we've had too many NaN epochs, reduce learning rate or stop
            if consecutive_nan_epochs >= 3:
                epochs_progress.write("Too many consecutive NaN epochs, reducing learning rate")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                
                if optimizer.param_groups[0]['lr'] < 1e-6:
                    epochs_progress.write("Learning rate too small, stopping training")
                    break
                    
                consecutive_nan_epochs = 0
                
            continue
        else:
            consecutive_nan_epochs = 0
        
        # Evaluate on dev set
        dev_loss, dev_mae, dev_rmse, dev_r2 = evaluate(model, dev_loader, criterion, device)
        
        # Learning rate scheduler
        scheduler.step(dev_loss)
        
        # Save checkpoints periodically
        checkpoint_interval = getattr(args, 'checkpoint_interval', 10)
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'dev_loss': dev_loss,
            }, checkpoint_path)
            epochs_progress.write(f"  Checkpoint saved at epoch {epoch+1}")
        
        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_progress.write(f"  New best model saved with dev loss: {dev_loss:.4f}")
        
        # Log metrics
        elapsed_time = time.time() - start_time
        
        # Update progress bar description with metrics
        epochs_progress.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'dev_loss': f"{dev_loss:.4f}",
            'dev_mae': f"{dev_mae:.4f}"
        })
        
        # Write detailed metrics to console
        epochs_progress.write(f"Epoch {epoch+1}/{args.num_epochs}:")
        epochs_progress.write(f"  Train loss: {train_loss:.4f}")
        epochs_progress.write(f"  Dev loss: {dev_loss:.4f}, MAE: {dev_mae:.4f}, RMSE: {dev_rmse:.4f}, R²: {dev_r2:.4f}")
        epochs_progress.write(f"  Time: {elapsed_time:.2f}s")
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/dev", dev_loss, epoch)
        writer.add_scalar("Metrics/dev_mae", dev_mae, epoch)
        writer.add_scalar("Metrics/dev_rmse", dev_rmse, epoch)
        writer.add_scalar("Metrics/dev_r2", dev_r2, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
    
    # Evaluate on test set using best model
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_mae, test_rmse, test_r2 = evaluate(model, test_loader, criterion, device)
    
    print("\nTest Results:")
    print(f"  Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    # Save final metrics
    metrics = {
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }
    
    writer.add_hparams(
        {"hidden_size": args.hidden_size,
         "num_layers": args.num_layers,
         "num_blocks": args.num_blocks,
         "lstm_type": args.lstm_type,
         "dropout": args.dropout,
         "lr": args.learning_rate},
        metrics
    )
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depression severity prediction using xLSTM")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the E-DAIC dataset")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model and results")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size of the LSTM")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of LSTM layers per block")
    parser.add_argument("--num_blocks", type=int, default=2,
                        help="Number of xLSTM blocks")
    parser.add_argument("--lstm_type", type=str, default="slstm", choices=["slstm", "mlstm"],
                        help="Type of LSTM to use")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--max_seq_length", type=int, default=1000,
                        help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=500,
                        help="Stride for sequence sampling")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")
    parser.add_argument("--gradient_clip", type=float, default=None,
                        help="Value to clip gradients to (default: no clipping)")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Interval (in epochs) to save checkpoints")
    
    args = parser.parse_args()
    train(args)
