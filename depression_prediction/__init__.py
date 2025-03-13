"""
Depression severity prediction using xLSTM with pose, gaze, and AU features.

This package provides functionality to predict depression severity using 
the PHQ-8 scale from pose, gaze, and action unit (AU) features extracted
from the E-DAIC dataset.
"""

from .data_loader import DepressionDataset, get_data_loaders
from .models import DepressionPredictor, create_model
from .utils import set_seed, get_feature_dimension, plot_predictions, compute_confusion_matrix

__all__ = [
    "DepressionDataset",
    "get_data_loaders",
    "DepressionPredictor",
    "create_model",
    "set_seed",
    "get_feature_dimension",
    "plot_predictions",
    "compute_confusion_matrix"
]
