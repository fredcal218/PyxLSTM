import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
import os
import json
import gc
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_fscore_support

from trainer import calculate_regression_metrics, calculate_classification_metrics

class EnsembleLoss(nn.Module):
    """
    Custom loss function for the ensemble model that combines:
    1. Binary cross-entropy for the classifier
    2. Weighted MAE for the regressors, with weighting based on which regressor was used
    """
    def __init__(self, classifier_weight=0.5, regressor_weight=1.0, scale_factor=0.2):
        super(EnsembleLoss, self).__init__()
        self.classifier_weight = classifier_weight
        self.regressor_weight = regressor_weight
        self.scale_factor = scale_factor
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions, targets):
        """
        Calculate combined loss.
        
        Args:
            predictions (tuple): Tuple of (scores, depression_prob, regressor_mask)
            targets (torch.Tensor): True PHQ-8 scores
            
        Returns:
            torch.Tensor: Combined loss
        """
        scores, depression_prob, regressor_mask = predictions
        
        # Convert tensors to float32 to ensure consistent dtypes
        scores = scores.float()
        depression_prob = depression_prob.float()
        targets = targets.float()
        
        # Check for NaN or Inf values
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("WARNING: NaN or Inf values detected in scores. Replacing with zeros.")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            
        if torch.isnan(depression_prob).any() or torch.isinf(depression_prob).any():
            print("WARNING: NaN or Inf values detected in probabilities. Replacing with safe values.")
            depression_prob = torch.nan_to_num(depression_prob, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Explicitly clamp depression_prob to [0, 1] to prevent BCE assertion errors
        # This ensures all values passed to BCE are in valid range
        depression_prob = torch.clamp(depression_prob, min=0.0, max=1.0)
        
        # Binary targets: 1 if PHQ-8 >= 10, 0 otherwise
        binary_targets = (targets >= 10.0).float()
        
        # For extra safety, confirm binary_targets are properly in {0,1}
        binary_targets = torch.round(binary_targets)  # Ensure exactly 0 or 1
        
        # Binary classification loss - with extra safety checks
        try:
            classification_loss = self.bce_loss(depression_prob, binary_targets)
        except Exception as e:
            print(f"WARNING: Error in BCE loss calculation: {str(e)}")
            print(f"Depression prob stats: min={depression_prob.min().item()}, max={depression_prob.max().item()}")
            print(f"Binary targets stats: min={binary_targets.min().item()}, max={binary_targets.max().item()}")
            # Fallback: use MSE loss instead if BCE fails
            classification_loss = nn.functional.mse_loss(depression_prob, binary_targets)
        
        # Regression loss with weighting based on PHQ-8 score
        regression_loss = torch.abs(scores - targets)
        
        # Weight loss higher for higher PHQ-8 scores to address class imbalance
        weights = 1.0 + self.scale_factor * targets
        weighted_regression_loss = (regression_loss * weights).mean()
        
        # Combined loss
        total_loss = (self.classifier_weight * classification_loss + 
                      self.regressor_weight * weighted_regression_loss)
        
        # Final check for numerical stability
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("WARNING: Loss is NaN or Inf. Returning zero loss to prevent training failure.")
            return torch.tensor(0.0, device=total_loss.device, requires_grad=True)
        
        return total_loss

class EnsembleTrainer:
    """Trainer for the ensemble depression prediction model"""
    
    def __init__(self, model, train_dataset, val_dataset, batch_size, learning_rate, save_dir, gradient_accumulation_steps=1):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize losses
        self.criterion = EnsembleLoss(classifier_weight=0.5, regressor_weight=1.0)
        
        # Setup specialized datasets
        train_low_indices, train_high_indices = self._split_dataset_by_severity(train_dataset)
        
        # Create subsets
        train_low_subset = Subset(train_dataset, train_low_indices)
        train_high_subset = Subset(train_dataset, train_high_indices)
        
        # Setup optimizers
        self.classifier_optimizer = optim.Adam(
            self.model.binary_classifier.parameters(), 
            lr=learning_rate
        )
        
        self.low_regressor_optimizer = optim.Adam(
            self.model.low_regressor.parameters(), 
            lr=learning_rate
        )
        
        self.high_regressor_optimizer = optim.Adam(
            self.model.high_regressor.parameters(), 
            lr=learning_rate
        )
        
        # Create DataLoaders with stratified sampling for training
        self.train_loader = self._create_balanced_dataloader(train_dataset)
        self.train_low_loader = DataLoader(train_low_subset, batch_size=batch_size, shuffle=True)
        self.train_high_loader = DataLoader(train_high_subset, batch_size=batch_size, shuffle=True)
        
        # Regular DataLoader for validation
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Print dataset statistics
        print(f"Training set: {len(train_dataset)} samples total")
        print(f"  Low range (0-9): {len(train_low_indices)} samples")
        print(f"  High range (10-24): {len(train_high_indices)} samples")
        print(f"Using gradient accumulation with {gradient_accumulation_steps} steps " +
              f"(effective batch size = {batch_size * gradient_accumulation_steps})")
        
        # Create directory for saving models
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def _split_dataset_by_severity(self, dataset):
        """Split dataset indices into low and high severity groups"""
        low_indices = []  # PHQ-8 scores 0-9
        high_indices = []  # PHQ-8 scores 10-24
        
        for idx, (_, label) in enumerate(dataset):
            score = label.item()
            if score < 10:
                low_indices.append(idx)
            else:
                high_indices.append(idx)
        
        return low_indices, high_indices
    
    def _create_balanced_dataloader(self, dataset):
        """Create a dataloader with balanced sampling across severity levels"""
        # Extract labels
        labels = np.array(dataset.labels)
        
        # Convert PHQ-8 scores to severity levels (0-4)
        severity_levels = np.digitize(labels, bins=[0, 5, 10, 15, 20, 25]) - 1
        
        # Count samples in each severity level
        level_counts = np.bincount(severity_levels)
        
        # Calculate weights (inverse of frequency)
        weights = 1.0 / (level_counts[severity_levels] + 1e-8)
        
        # Create and return sampler and loader
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
    
    def train(self, epochs, patience):
        """Train the ensemble model"""
        best_val_mae = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        # Initialize history dictionary
        history = {
            'train_loss': [], 'train_classifier_loss': [], 'train_low_loss': [], 'train_high_loss': [],
            'train_mae': [], 'train_rmse': [], 'train_r2': [], 
            'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': [], 
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }
        
        print(f"Starting ensemble training on device: {self.device}")
        print(f"Initial learning rate: {self.learning_rate}")
        
        for epoch in range(epochs):
            # Phase 1: Train the binary classifier with gradient accumulation
            classifier_loss = self._train_classifier_with_accumulation(self.train_loader)
            
            # Phase 2: Train the low-range regressor with gradient accumulation
            low_loss = self._train_low_regressor_with_accumulation(self.train_low_loader)
            
            # Phase 3: Train the high-range regressor with gradient accumulation
            high_loss = self._train_high_regressor_with_accumulation(self.train_high_loader)
            
            # Clear memory between phases
            torch.cuda.empty_cache()
            gc.collect()
            
            # Phase 4: Evaluate the full ensemble on all data with memory management
            train_metrics = self.evaluate_with_memory_management(self.train_loader, is_training=True)
            val_metrics = self.evaluate_with_memory_management(self.val_loader, is_training=False)
            
            # Unpack metrics
            train_loss, train_mae, train_rmse, train_r2, train_precision, train_recall, train_f1 = train_metrics
            val_loss, val_mae, val_rmse, val_r2, val_precision, val_recall, val_f1 = val_metrics
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['train_classifier_loss'].append(classifier_loss)
            history['train_low_loss'].append(low_loss)
            history['train_high_loss'].append(high_loss)
            history['train_mae'].append(train_mae)
            history['train_rmse'].append(train_rmse)
            history['train_r2'].append(train_r2)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            history['val_r2'].append(val_r2)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            
            # Print epoch metrics
            print(f"Epoch [{epoch+1}/{epochs}]:")
            print(f"  Train - Losses: Full={train_loss:.4f}, Classifier={classifier_loss:.4f}, " +
                  f"Low={low_loss:.4f}, High={high_loss:.4f}")
            print(f"        - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
            print(f"        - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
            print(f"        - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            # Early stopping check based on validation MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"  New best model! Saving checkpoint (val_mae: {val_mae:.4f})")
                self._save_model()
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter}/{patience} epochs (best val_mae: {best_val_mae:.4f} @ epoch {best_epoch})")
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                print(f"Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
                break
        
        # Save training history
        self._save_history(history)
        
        print(f"Training completed after {len(history['train_loss'])} epochs")
        print(f"Best model saved at epoch {best_epoch} with validation MAE: {best_val_mae:.4f}")
        
        return history
    
    def _train_classifier_with_accumulation(self, data_loader):
        """Train the binary classifier with gradient accumulation"""
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        # Binary cross entropy loss
        criterion = nn.BCELoss()
        self.classifier_optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Binary labels: 1 if PHQ-8 >= 10, 0 otherwise
            binary_targets = (targets >= 10.0).float()
            
            # Forward pass through classifier only
            logits = self.model.binary_classifier(inputs)
            probs = self.model.sigmoid(logits)
            
            # Calculate loss
            loss = criterion(probs, binary_targets)
            # Normalize loss to account for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size * self.gradient_accumulation_steps  # Scale back to get actual loss
            total_samples += batch_size
            
            # Perform optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                self.classifier_optimizer.step()
                self.classifier_optimizer.zero_grad()
                
                # Clear memory
                torch.cuda.empty_cache()
        
        return running_loss / total_samples
    
    def _train_low_regressor_with_accumulation(self, data_loader):
        """Train the low-range regressor with gradient accumulation"""
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        # Mean absolute error
        criterion = nn.L1Loss()
        self.low_regressor_optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass through low regressor only
            predictions = self.model.low_regressor(inputs)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            # Normalize loss to account for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size * self.gradient_accumulation_steps  # Scale back
            total_samples += batch_size
            
            # Perform optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                self.low_regressor_optimizer.step()
                self.low_regressor_optimizer.zero_grad()
                
                # Clear memory
                torch.cuda.empty_cache()
        
        return running_loss / total_samples
    
    def _train_high_regressor_with_accumulation(self, data_loader):
        """Train the high-range regressor with gradient accumulation"""
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        # Mean absolute error
        criterion = nn.L1Loss()
        self.high_regressor_optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass through high regressor only
            predictions = self.model.high_regressor(inputs)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            # Normalize loss to account for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size * self.gradient_accumulation_steps  # Scale back
            total_samples += batch_size
            
            # Perform optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                self.high_regressor_optimizer.step()
                self.high_regressor_optimizer.zero_grad()
                
                # Clear memory
                torch.cuda.empty_cache()
        
        return running_loss / total_samples
    
    def evaluate_with_memory_management(self, data_loader, is_training=False):
        """Evaluate the ensemble model with memory management strategies"""
        phase = "Training" if is_training else "Validation"
        print(f"\n--- Starting {phase} Evaluation ---")
        eval_start = time.time()
        
        if is_training:
            self.model.train()
            print("Model in training mode")
        else:
            self.model.eval()
            print("Model in evaluation mode")
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        total_samples = 0
        
        # Process batches with memory management
        print(f"Processing {len(data_loader)} batches for {phase.lower()} evaluation")
        with torch.set_grad_enabled(is_training):
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                batch_start = time.time()
                print(f"\nBatch {batch_idx+1}/{len(data_loader)} - {phase}")
                
                # Extra diagnostic for batch 7
                if batch_idx == 6:  # 0-indexed, so batch 7 is index 6
                    print(f"*** DETAILED DIAGNOSTICS FOR BATCH 7 ***")
                    print(f"Targets shape: {targets.shape}, min: {targets.min()}, max: {targets.max()}")
                    print(f"Inputs shape: {inputs.shape}, has NaN: {torch.isnan(inputs).any()}, has Inf: {torch.isinf(inputs).any()}")
                
                # Move data to device
                print(f"  Moving batch to {self.device}...")
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass through the ensemble
                print(f"  Running model forward pass...")
                predictions = self.model(inputs)
                
                # Extra diagnostics for batch 7
                if batch_idx == 6:
                    print("Prediction stats:")
                    print(f"  Scores: min={predictions[0].min().item()}, max={predictions[0].max().item()}")
                    print(f"  Probs: min={predictions[1].min().item()}, max={predictions[1].max().item()}")
                
                print(f"  Calculating loss...")
                try:
                    loss = self.criterion(predictions, targets)
                    
                    # Store predictions and targets on CPU to save GPU memory
                    print(f"  Moving results to CPU...")
                    all_predictions.append(predictions[0].detach().cpu())
                    all_targets.append(targets.cpu())
                    
                    # Update metrics
                    batch_size = targets.size(0)
                    running_loss += loss.item() * batch_size
                    total_samples += batch_size
                except Exception as e:
                    print(f"ERROR in batch {batch_idx+1}: {str(e)}")
                    print("Skipping this batch and continuing...")
                    continue
                
                # Clear memory between batches
                del inputs, predictions, loss
                torch.cuda.empty_cache()
                
                batch_time = time.time() - batch_start
                print(f"  ✓ Batch {batch_idx+1} processed in {batch_time:.2f}s")
                
                # Show memory usage if on CUDA
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(f"  CUDA Memory: {allocated:.3f} GB (max: {max_allocated:.3f} GB)")
        
        print(f"Collecting and processing all results...")
        # Move to CPU for final processing
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        print(f"Calculating metrics...")
        # Calculate metrics
        rmse, mae, r2 = calculate_regression_metrics(all_predictions, all_targets)
        precision, recall, f1 = calculate_classification_metrics(all_predictions, all_targets)
        
        eval_time = time.time() - eval_start
        print(f"✓ {phase} evaluation completed in {eval_time:.2f}s")
        return running_loss / total_samples, mae, rmse, r2, precision, recall, f1
    
    def _save_model(self):
        """Save the ensemble model"""
        model_path = os.path.join(self.save_dir, 'best_ensemble.pt')
        
        torch.save({
            'binary_classifier': self.model.binary_classifier.state_dict(),
            'low_regressor': self.model.low_regressor.state_dict(),
            'high_regressor': self.model.high_regressor.state_dict(),
            'classifier_optimizer': self.classifier_optimizer.state_dict(),
            'low_optimizer': self.low_regressor_optimizer.state_dict(),
            'high_optimizer': self.high_regressor_optimizer.state_dict(),
            'confidence_threshold': self.model.confidence_threshold
        }, model_path)
    
    def _save_history(self, history):
        """Save the training history"""
        history_path = os.path.join(self.save_dir, 'ensemble_history.json')
        
        # Convert values to Python native types for JSON serialization
        history_serializable = {
            k: [float(val) for val in v] for k, v in history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
            
        print(f"Training history saved to {history_path}")
