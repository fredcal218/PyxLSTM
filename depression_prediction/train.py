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
import matplotlib.pyplot as plt

from depression_prediction.data_loader import get_data_loaders
from depression_prediction.models import create_model
from depression_prediction.utils import set_seed, get_feature_dimension
from depression_prediction.sequence_utils import process_long_sequence
from depression_prediction.utils import plot_evaluation_results

# Import for mixed precision training
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Mixed precision training not available")


def train_epoch(model, data_loader, criterion, optimizer, device, scaler=None, gradient_clip=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    batch_count = 0
    nan_batch_count = 0
    
    # Add progress bar for batches
    progress_bar = tqdm(data_loader, desc="Training batches", leave=False)
    
    for features, labels in progress_bar:
        # Skip empty batches
        if features.size(0) == 0:
            continue
            
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Extra guard against NaN inputs
        if torch.isnan(features).any() or torch.isinf(features).any():
            nan_batch_count += 1
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)  # Slightly more efficient
        
        # Use mixed precision if available
        if scaler is not None:
            with autocast('cuda'):
                outputs, _ = model(features)
                loss = criterion(outputs, labels)
                
            # Check for NaN loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                nan_batch_count += 1
                progress_bar.set_postfix(loss="NaN/Inf")
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
                nan_batch_count += 1
                progress_bar.set_postfix(loss="NaN/Inf")
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
        batch_count += 1
        
        # Free memory explicitly
        del features, labels, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Update progress bar description with current loss
        progress_bar.set_postfix(loss=f"{current_loss:.4f}")
    
    # Check if all batches were NaNs
    if batch_count == 0:
        return float('nan'), nan_batch_count
    
    # Return average loss and NaN count
    return total_loss / batch_count, nan_batch_count


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    batch_count = 0
    all_preds = []
    all_labels = []
    
    # Add progress bar for evaluation
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for features, labels in progress_bar:
            # Skip empty batches
            if features.size(0) == 0:
                continue
                
            labels = labels.to(device, non_blocking=True)
            
            # Check if sequence is too long for direct processing
            if features.size(1) * features.size(2) > 100000000:  # Threshold for "very long" sequences
                # Process in chunks
                outputs, _ = process_long_sequence(model, features, max_chunk_length=5000, 
                                                  overlap=1000, device=device)
            else:
                # Standard processing
                features = features.to(device, non_blocking=True)
                # Safety check for NaN inputs
                if torch.isnan(features).any() or torch.isinf(features).any():
                    features = torch.nan_to_num(features)
                outputs, _ = model(features)
            
            loss = criterion(outputs, labels)
            
            # Check for NaN loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue
                
            total_loss += loss.item()
            batch_count += 1
            
            # Make sure outputs are clean for metrics
            clean_outputs = torch.nan_to_num(outputs).cpu().numpy()
            all_preds.extend(clean_outputs)
            all_labels.extend(labels.cpu().numpy())
            
            # Free memory
            del features, labels, outputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Handle case where all batches failed
    if batch_count == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    
    # Calculate evaluation metrics
    try:
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        r2 = r2_score(all_labels, all_preds)
    except:
        # Handle case where metrics calculation fails
        mae, rmse, r2 = float('nan'), float('nan'), float('nan')
    
    return total_loss / batch_count, mae, rmse, r2


def predict_with_best_model(model, data_loader, device, model_path):
    """
    Make predictions using the best saved model.
    
    Args:
        model: Model to use for predictions
        data_loader: Data loader containing test data
        device: Device to run predictions on
        model_path: Path to the best model weights
        
    Returns:
        tuple: Predictions and true labels
    """
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc="Predicting", leave=False)
    
    with torch.no_grad():
        for features, labels in progress_bar:
            if features.size(0) == 0:
                continue
                
            # Handle very long sequences using chunking
            if features.size(1) * features.size(2) > 100000000:
                outputs, _ = process_long_sequence(model, features, max_chunk_length=5000, 
                                                  overlap=1000, device=device)
            else:
                features = features.to(device, non_blocking=True)
                if torch.isnan(features).any() or torch.isinf(features).any():
                    features = torch.nan_to_num(features)
                outputs, _ = model(features)
            
            all_preds.extend(torch.nan_to_num(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels


def train(args):
    """Train and evaluate the model."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Set up device - Force CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Optional: Set this to optimize for your specific GPU
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: No GPU detected. Running on CPU will be significantly slower!")
    
    # Load data
    print("\n=== Loading data ===")
    start_time = time.time()
    num_workers = getattr(args, 'num_workers', 0)
    train_loader, dev_loader, test_loader = get_data_loaders(
        args.data_dir, args.batch_size, args.max_seq_length, args.stride, num_workers
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
    criterion = nn.SmoothL1Loss()  # Huber loss with better stability properties
    print(f"Loss function: SmoothL1Loss (Huber loss)")
    
    # Optimizer with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.005)
    print(f"Optimizer: AdamW with learning rate {args.learning_rate} and weight decay 0.005")
    
    # Learning rate scheduler with warmup
    def lr_lambda(current_step):
        warmup_steps = 5
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print("Scheduler: LambdaLR with warmup")
    
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
    best_dev_mae = float('inf')
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    consecutive_nan_epochs = 0
    
    # Variables to track training progress
    start_epoch = 0
    
    # Resume training from checkpoint if specified
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
        if hasattr(args, 'checkpoint_path') and os.path.isfile(args.checkpoint_path):
            print(f"\n=== Resuming from checkpoint: {args.checkpoint_path} ===")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            
            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if it exists
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Update start epoch
            start_epoch = checkpoint['epoch'] + 1
            
            # Update best metrics if available
            if 'dev_loss' in checkpoint:
                best_dev_loss = checkpoint['dev_loss']
            
            if 'dev_mae' in checkpoint:
                best_dev_mae = checkpoint['dev_mae']
                
            print(f"Resuming training from epoch {start_epoch}")
            print(f"Previous best dev loss: {best_dev_loss:.4f}")
            print(f"Previous best dev MAE: {best_dev_mae:.4f}")
        else:
            print(f"Warning: Checkpoint file not found at {getattr(args, 'checkpoint_path', 'unknown path')}")
            print("Starting training from scratch.")
    
    # Early stopping setup
    early_stopping = getattr(args, 'early_stopping', False)
    early_stopping_patience = getattr(args, 'early_stopping_patience', 5)
    early_stopping_metric = getattr(args, 'early_stopping_metric', 'mae')
    no_improvement_count = 0
    best_metric_value = float('inf')  # For metrics where lower is better (loss, mae, rmse)
    
    print(f"Early stopping: {'Enabled' if early_stopping else 'Disabled'}")
    if early_stopping:
        print(f"  Patience: {early_stopping_patience} epochs")
        print(f"  Early Stopping Metric: {early_stopping_metric}")
    
    # Create progress bar for epochs
    epochs_progress = tqdm(range(start_epoch, args.num_epochs), desc="Training epochs", unit="epoch")
    
    for epoch in epochs_progress:
        start_time = time.time()
        
        # Train
        train_loss, nan_batch_count = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            scaler=scaler, gradient_clip=gradient_clip
        )
        
        # Check for NaN in training loss
        if np.isnan(train_loss) or np.isinf(train_loss):
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
            
        # Report NaN batch statistics
        if nan_batch_count > 0:
            epochs_progress.write(f"  {nan_batch_count} batches with NaN/Inf were skipped")
        
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
                'dev_mae': dev_mae,
            }, checkpoint_path)
            epochs_progress.write(f"  Checkpoint saved at epoch {epoch+1}")
        
        # Save best model based on dev loss
        if not np.isnan(dev_loss) and dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_progress.write(f"  New best model saved with dev loss: {dev_loss:.4f}")
        
        # Save best model based on dev MAE (useful for regression tasks)
        if not np.isnan(dev_mae) and dev_mae < best_dev_mae:
            best_dev_mae = dev_mae
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_mae_model.pt"))
            epochs_progress.write(f"  New best MAE model saved with dev MAE: {dev_mae:.4f}")
        
        # Early stopping check
        if early_stopping:
            # Get current metric value based on the selected metric
            if early_stopping_metric == 'loss':
                current_metric = dev_loss
            elif early_stopping_metric == 'mae':
                current_metric = dev_mae
            elif early_stopping_metric == 'rmse':
                current_metric = dev_rmse
            elif early_stopping_metric == 'r2':
                current_metric = -dev_r2  # Negative because for R2, higher is better
            
            # Check if there's improvement
            if np.isnan(current_metric) or current_metric >= best_metric_value:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    epochs_progress.write(f"\nEarly stopping triggered! No improvement in {early_stopping_metric} "
                                         f"for {early_stopping_patience} epochs.")
                    break
            else:
                # There is improvement
                best_metric_value = current_metric
                no_improvement_count = 0
                epochs_progress.write(f"  New best {early_stopping_metric}: {current_metric:.4f}")
        
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
    
    # Check if we have a valid best model
    if not os.path.exists(best_model_path):
        print("\nNo valid model was saved during training. Saving the final model instead.")
        torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
        best_model_path = os.path.join(args.output_dir, "final_model.pt")
    
    # Final evaluation section
    print("\n=== Evaluating best model on test set ===")
    if not os.path.exists(best_model_path):
        print("No best model found. Using final model.")
        best_model_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save(model.state_dict(), best_model_path)
    
    # Make predictions with the best model
    test_preds, test_labels = predict_with_best_model(model, test_loader, device, best_model_path)
    
    # Calculate metrics and generate evaluation plots
    test_metrics = plot_evaluation_results(
        test_labels, test_preds, args.output_dir, prefix="test_"
    )
    
    # Calculate loss separately since it's not part of the plotting function
    test_loss = criterion(torch.tensor(test_preds), torch.tensor(test_labels)).item()
    
    print("\nTest Results:")
    print(f"  Loss: {test_loss:.4f}, MAE: {test_metrics['mae']:.4f}, " + 
          f"RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
    
    # Also evaluate best MAE model if it's different from the best loss model
    mae_model_path = os.path.join(args.output_dir, "best_mae_model.pt")
    if os.path.exists(mae_model_path) and mae_model_path != best_model_path:
        print("\n=== Evaluating best MAE model on test set ===")
        mae_preds, mae_labels = predict_with_best_model(model, test_loader, device, mae_model_path)
        
        mae_metrics = plot_evaluation_results(
            mae_labels, mae_preds, args.output_dir, prefix="mae_model_"
        )
        
        print("\nBest MAE Model Test Results:")
        print(f"  MAE: {mae_metrics['mae']:.4f}, RMSE: {mae_metrics['rmse']:.4f}, R²: {mae_metrics['r2']:.4f}")
    
    writer.close()
    print("\nTraining and evaluation completed!")


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
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Patience for early stopping")
    parser.add_argument("--early_stopping_metric", type=str, default="mae", choices=["loss", "mae", "rmse", "r2"],
                        help="Metric for early stopping")
    parser.add_argument("--resume_from_checkpoint", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to the checkpoint file to resume from")
    
    args = parser.parse_args()
    train(args)
