# Depression Binary Classification Model

This module implements a binary classification model for depression detection using the E-DAIC dataset, with a focus on feature importance and interpretability.

## Features

- Binary classification using direct PHQ_Binary labels
- Feature importance analysis using integrated gradients
- Attention visualization from xLSTM architecture
- Global and instance-specific feature importance
- Clinical AU analysis based on depression literature

## Files

- `model.py` - Binary classification model with interpretability features
- `data_processor.py` - Data loading and preprocessing for E-DAIC dataset
- `train.py` - Training script with evaluation and visualization

## Usage

1. Ensure the E-DAIC dataset is available in the expected directory structure
2. Run the training script:

```bash
cd model_binary
python train.py
```

Training results, feature importance visualizations, and model metrics will be saved to the `results/binary_classification_{timestamp}` directory.

## Expected Directory Structure

```
E-DAIC/
├── data_extr/
│   ├── train/
│   │   ├── {participant_id}_P/
│   │   │   └── features/
│   │   │       └── {participant_id}_OpenFace2.1.0_Pose_Gaze_AUs.csv
│   ├── dev/
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train_split.csv  # Contains Participant_ID and PHQ_Binary columns
    ├── dev_split.csv
    └── test_split.csv
```

## Expected Label Format

The label CSV files should contain these columns:
- `Participant_ID`: Unique identifier for each participant
- `PHQ_Binary`: Binary label for depression (1 = depressed, 0 = not depressed)
- `PHQ8_Score` (optional): Original PHQ-8 scores for reference

## Interpretability

This model provides several ways to interpret which features are important for depression classification:

1. **Global Feature Importance** - Shows which features are generally important across all participants
2. **Clinical AU Analysis** - Focuses on clinically significant Action Units from depression literature
3. **Instance-specific Importance** - Shows which features influenced a specific participant's classification
4. **Comparison Visualization** - Compares important features between depressed and non-depressed examples

## Hyperparameters

- `BATCH_SIZE` = 16
- `HIDDEN_SIZE` = 128
- `NUM_LAYERS` = 2
- `DROPOUT` = 0.5
- `MAX_SEQ_LENGTH` = 150
- `LEARNING_RATE` = 0.001
- `NUM_EPOCHS` = 50
- `EARLY_STOPPING_PATIENCE` = 10
