import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime
import time
from tqdm import tqdm
import copy
import warnings

class CrossValidationTrainer:
    def __init__(self, model_class, train_dataset, val_dataset, test_dataset, params):
        """
        Initialize the CrossValidationTrainer.
        
        Args:
            model_class: The BiLSTM_Attention model class
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            params: Dict containing training parameters and hyperparameters
        """
        self.model_class = model_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.params = params
        
        # Create combined dataset for cross-validation
        self.combined_dataset = ConcatDataset([self.train_dataset, self.val_dataset])
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory
        self.save_dir = params.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_model(self, model, train_loader, criterion, optimizer, epochs, max_grad_norm=1.0):
        """
        Train a model for specified epochs with gradient clipping
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of epochs to train for
            max_grad_norm: Maximum norm for gradient clipping
            
        Returns:
            Trained model and flag indicating if training was successful
        """
        model.train()
        training_successful = True
        
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Check for NaN in loss
                if torch.isnan(loss).any():
                    print(f"  Warning: NaN loss detected at epoch {epoch+1}, batch {batch_count+1}")
                    print(f"  Training will continue but results may be unreliable")
                    running_loss = float('nan')
                    training_successful = False
                    continue
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
            
            # Print epoch loss every 5 epochs
            if (epoch + 1) % 5 == 0:
                avg_loss = running_loss / max(1, batch_count)
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                
                # Early stopping if loss is NaN
                if np.isnan(avg_loss):
                    print(f"  Stopping training early due to NaN loss")
                    training_successful = False
                    break
                
        return model, training_successful
    
    def evaluate_model(self, model, data_loader, criterion):
        """
        Evaluate model on data loader with NaN handling
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data
            criterion: Loss function
            
        Returns:
            Dict containing evaluation metrics
        """
        model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all outputs and targets
        outputs = torch.cat(all_outputs, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Check for NaN in outputs
        if np.isnan(outputs).any():
            print("  Warning: NaN values detected in model outputs during evaluation")
            print("  Using fallback metrics")
            return {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan')
            }
        
        # Calculate metrics with error handling
        try:
            mse = mean_squared_error(targets, outputs)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, outputs)
            r2 = r2_score(targets, outputs)
        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            return {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan')
            }
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def perform_cross_validation(self, param_grid=None, n_folds=5, inner_folds=5):
        """
        Perform nested cross-validation with hyperparameter tuning.
        
        Args:
            param_grid: Dict of hyperparameter names and values to search
            n_folds: Number of outer folds for CV
            inner_folds: Number of inner folds for hyperparameter tuning
            
        Returns:
            Dict of cross-validation results
        """
        # Default param grid if none provided
        if param_grid is None:
            param_grid = {
                'hidden_size': [32, 64, 128],
                'num_layers': [2, 3],
                'dropout': [0.2, 0.3, 0.4],
                'learning_rate': [0.01, 0.05, 0.1]  # Added intermediate value
            }
        
        # Define outer cross-validation
        outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Get indices for all samples in combined dataset
        indices = list(range(len(self.combined_dataset)))
        
        # Store results
        outer_results = []
        best_models = []
        best_params_per_fold = []
        
        # Perform outer cross-validation
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(indices)):
            print(f"\nOuter Fold {fold+1}/{n_folds}")
            
            # Create train and validation datasets for this fold
            outer_train_dataset = Subset(self.combined_dataset, train_idx)
            outer_val_dataset = Subset(self.combined_dataset, val_idx)
            
            # Find best hyperparameters using inner cross-validation
            best_params, best_score = self._find_best_params(
                outer_train_dataset, param_grid, inner_folds
            )
            
            print(f"Best parameters found: {best_params} with validation RMSE: {best_score:.4f}")
            best_params_per_fold.append(best_params)
            
            # Train model with best parameters on all training data
            train_loader = DataLoader(
                outer_train_dataset, 
                batch_size=self.params['batch_size'], 
                shuffle=True
            )
            val_loader = DataLoader(
                outer_val_dataset, 
                batch_size=self.params['batch_size']
            )
            
            # Initialize model with best parameters
            model = self._create_model(**best_params)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
            
            # Train model
            print(f"Training final model for fold {fold+1} with best parameters")
            model, training_successful = self.train_model(
                model, 
                train_loader, 
                criterion, 
                optimizer, 
                self.params['epochs'],
                max_grad_norm=1.0  # Apply gradient clipping
            )
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model(model, val_loader, criterion)
            print(f"Validation RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
            
            # Save model
            model_path = os.path.join(self.save_dir, f"model_fold{fold+1}.pt")
            torch.save(model.state_dict(), model_path)
            best_models.append(model.state_dict())
            
            # Store results
            result = {
                'fold': fold + 1,
                'best_params': best_params,
                'validation_metrics': val_metrics,
                'training_successful': training_successful
            }
            outer_results.append(result)
        
        # Evaluate ensemble on test set
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.params['batch_size']
        )
        
        test_metrics = self._evaluate_ensemble(best_models, best_params_per_fold, test_loader)
        
        # Prepare results dictionary
        results = {
            'outer_cv_results': outer_results,
            'best_params_per_fold': best_params_per_fold,
            'test_metrics': test_metrics
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _find_best_params(self, dataset, param_grid, n_folds):
        """
        Find best hyperparameters using inner cross-validation.
        
        Args:
            dataset: Dataset to use for parameter tuning
            param_grid: Dict of hyperparameter names and values to search
            n_folds: Number of inner folds
            
        Returns:
            Tuple (best_params, best_score)
        """
        # Define inner cross-validation
        inner_cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Get indices for all samples in dataset
        indices = list(range(len(dataset)))
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        print(f"Searching {len(param_combinations)} parameter combinations")
        
        best_params = None
        best_avg_score = float('inf')  # Lower RMSE is better
        
        # Try each parameter combination
        for param_idx, params in enumerate(param_combinations):
            print(f"Parameter set {param_idx+1}/{len(param_combinations)}: {params}")
            
            fold_scores = []
            valid_fold_count = 0
            
            # Perform k-fold cross validation
            for k, (train_idx, val_idx) in enumerate(inner_cv.split(indices)):
                # Create train and validation datasets for this inner fold
                inner_train_dataset = Subset(dataset, train_idx)
                inner_val_dataset = Subset(dataset, val_idx)
                
                inner_train_loader = DataLoader(
                    inner_train_dataset, 
                    batch_size=self.params['batch_size'], 
                    shuffle=True
                )
                inner_val_loader = DataLoader(
                    inner_val_dataset, 
                    batch_size=self.params['batch_size']
                )
                
                # Initialize model with these parameters
                model = self._create_model(**params)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
                
                # Train model with gradient clipping
                model, training_successful = self.train_model(
                    model, 
                    inner_train_loader, 
                    criterion, 
                    optimizer, 
                    self.params['epochs'],
                    max_grad_norm=1.0  # Add gradient clipping
                )
                
                # Evaluate on validation set
                val_metrics = self.evaluate_model(model, inner_val_loader, criterion)
                
                # Check if we got valid results
                if not np.isnan(val_metrics['rmse']):
                    fold_scores.append(val_metrics['rmse'])
                    valid_fold_count += 1
                    print(f"  Fold {k+1}: RMSE = {val_metrics['rmse']:.4f}")
                else:
                    print(f"  Fold {k+1}: RMSE = NaN (skipping this fold)")
            
            # Calculate average score across folds, if we have any valid folds
            if valid_fold_count > 0:
                avg_score = np.mean(fold_scores)
                print(f"  Average RMSE: {avg_score:.4f} (from {valid_fold_count}/{n_folds} valid folds)")
                
                # Check if this is better than our current best
                if avg_score < best_avg_score:
                    best_avg_score = avg_score
                    best_params = params
                    print(f"  New best parameters found!")
            else:
                print(f"  No valid folds for this parameter set (all NaN)")
        
        # If we couldn't find valid parameters, use sensible defaults
        if best_params is None:
            print("Warning: No valid parameter combination found. Using default parameters.")
            best_params = {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.01  # Lower learning rate to avoid NaN
            }
            best_avg_score = float('inf')
            
        return best_params, best_avg_score
    
    def _create_model(self, hidden_size, num_layers, dropout, learning_rate):
        """Create model instance with specified parameters"""
        model = self.model_class(
            input_size=self.params['input_size'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_length=self.params['seq_length']
        )
        model.to(self.device)
        return model
    
    def _evaluate_ensemble(self, model_states, params_list, data_loader):
        """
        Evaluate ensemble of models on test set with NaN handling.
        
        Args:
            model_states: List of model state dicts
            params_list: List of parameter dicts for each model
            data_loader: Test data loader
            
        Returns:
            Dict of test metrics
        """
        print("\nEvaluating ensemble model on test set")
        
        all_targets = []
        all_preds = []
        valid_models = 0
        
        # For each model in the ensemble
        for i, (state_dict, params) in enumerate(zip(model_states, params_list)):
            # Create model instance
            model = self._create_model(**params)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Make predictions
            fold_preds = []
            has_nan = False
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(data_loader):
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    
                    # Check for NaN in outputs
                    if torch.isnan(outputs).any():
                        print(f"  Model {i+1} produced NaN outputs - excluding from ensemble")
                        has_nan = True
                        break
                    
                    fold_preds.append(outputs.cpu())
                    
                    # Store targets only once
                    if i == 0:
                        all_targets.append(targets)
            
            # Only include this model if it produced valid predictions
            if not has_nan and fold_preds:
                # Concatenate predictions from this model
                fold_preds = torch.cat(fold_preds, dim=0)
                all_preds.append(fold_preds)
                valid_models += 1
        
        # If we don't have any valid models, return NaN metrics
        if valid_models == 0:
            print("  No valid models in ensemble. Cannot evaluate.")
            return {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan')
            }
        
        # Average predictions from all valid models
        print(f"  Using {valid_models}/{len(model_states)} models for ensemble predictions")
        ensemble_preds = torch.mean(torch.stack(all_preds), dim=0).numpy()
        
        # Concatenate targets
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate metrics
        try:
            mse = mean_squared_error(targets, ensemble_preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, ensemble_preds)
            r2 = r2_score(targets, ensemble_preds)
            
            print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
        except Exception as e:
            print(f"  Error calculating ensemble metrics: {e}")
            return {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan')
            }
    
    def _generate_param_combinations(self, param_grid):
        """Generate all combinations of parameters from grid"""
        from itertools import product
        
        keys = list(param_grid.keys())
        values = [param_grid[key] for key in keys]
        combinations = []
        
        for combo in product(*values):
            param_dict = {k: v for k, v in zip(keys, combo)}
            combinations.append(param_dict)
            
        return combinations
    
    def _save_results(self, results):
        """Save cross-validation results to file"""
        results_path = os.path.join(self.save_dir, 'cv_results.json')
        
        # Add timestamp
        results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results['parameters'] = {
            k: v for k, v in self.params.items() 
            if k not in ('model_class', 'train_dataset', 'val_dataset', 'test_dataset')
        }
        
        # Make sure everything is JSON serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif np.isnan(obj):
                return "NaN"
            else:
                return obj
        
        serializable_results = make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to {results_path}")
