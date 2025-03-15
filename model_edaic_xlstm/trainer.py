import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time
import math
import json
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, learning_rate, save_dir):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Add learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,  # Reduce LR by half when plateauing
            patience=15,  # Wait for 15 epochs before reducing LR
            verbose=True, 
            min_lr=1e-6
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)

    def train(self, epochs, patience, seed):
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        history = {
            'train_loss': [], 'train_acc': [], 'train_rmse': [], 'train_mae': [], 'train_r2': [],
            'val_loss': [], 'val_acc': [], 'val_rmse': [], 'val_mae': [], 'val_r2': [],
            'learning_rates': []
        }

        print(f"Starting training on device: {self.device}")
        print(f"Initial learning rate: {self.learning_rate}")
        
        for epoch in range(epochs):
            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            train_metrics = train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.device)
            val_metrics = validate_epoch(self.model, self.val_loader, self.criterion, self.device)

            # Unpack metrics
            train_loss, train_acc, train_rmse, train_mae, train_r2 = train_metrics
            val_loss, val_acc, val_rmse, val_mae, val_r2 = val_metrics
            
            # Update learning rate scheduler based on validation loss
            self.scheduler.step(val_loss)
            
            # Store metrics in history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_rmse'].append(train_rmse)
            history['train_mae'].append(train_mae)
            history['train_r2'].append(train_r2)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_rmse'].append(val_rmse)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)

            print(f'Epoch [{epoch + 1}/{epochs}]: (LR: {current_lr:.6f})')
            print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}')
            print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"  New best model! Saving checkpoint (val_loss: {val_loss:.4f})")
                self.save_model()
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter}/{patience} epochs (best val_loss: {best_val_loss:.4f} @ epoch {best_epoch})")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break

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
        print(f"Best model was at epoch {best_epoch} with validation loss {best_val_loss:.4f}")
        
        return history

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        model_path = os.path.join(self.save_dir, 'best_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, model_path)
        print(f'Model saved to {model_path}')
        
    def load_best_model(self):
        """Load the best model from the saved checkpoint"""
        model_path = os.path.join(self.save_dir, 'best_model.pth')
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

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
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
        
        # Calculate batch accuracy
        batch_acc = model.calculate_accuracy(outputs, targets)
        
        # Store predictions and targets for metric calculation
        all_predictions.append(outputs.detach())
        all_targets.append(targets.detach())
        
        # Update metrics
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_acc += batch_acc * batch_size
        total_samples += batch_size
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate regression metrics
    rmse, mae, r2 = calculate_regression_metrics(predictions, targets)
    
    # Calculate final metrics
    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    
    return epoch_loss, epoch_acc, rmse, mae, r2

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_predictions = []
    all_targets = []
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate batch accuracy
            batch_acc = model.calculate_accuracy(outputs, targets)
            
            # Store predictions and targets for metric calculation
            all_predictions.append(outputs)
            all_targets.append(targets)
            
            # Update metrics
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_acc += batch_acc * batch_size
            total_samples += batch_size
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate regression metrics
    rmse, mae, r2 = calculate_regression_metrics(predictions, targets)
    
    # Calculate final metrics
    val_loss = running_loss / total_samples
    val_acc = running_acc / total_samples
    
    return val_loss, val_acc, rmse, mae, r2
