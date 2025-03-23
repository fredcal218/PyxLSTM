import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Configuration
DATA_DIR = "E-DAIC/data_extr"
LABELS_DIR = "E-DAIC/labels"
OUTPUT_DIR = "model_binary/feature_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of features to exclude from analysis (metadata and non-behavioral)
EXCLUDE_FEATURES = [
    'frame', 'face_id', 'timestamp', 'success', 
    'confidence', 'gaze_direction_0_', 'gaze_direction_1_'
]

def is_behavioral_feature(feature_name):
    """Check if a feature is behavioral (not metadata or confidence)"""
    for exclude in EXCLUDE_FEATURES:
        if exclude in feature_name:
            return False
    
    # Exclude confidence columns
    if feature_name.endswith('_c'):
        return False
    
    return True

def load_participants_data(split_path, data_dir, label_path):
    """
    Load all participants' data for a given split
    
    Args:
        split_path (str): Split directory ('train', 'dev', or 'test')
        data_dir (str): Base data directory
        label_path (str): Path to label CSV file
    
    Returns:
        tuple: (features_by_participant, labels)
    """
    # Load labels
    labels_df = pd.read_csv(label_path)
    
    # Map participant IDs to depression status
    depression_status = {}
    phq_scores = {}
    
    for _, row in labels_df.iterrows():
        pid = str(row['Participant_ID'])
        depression_status[pid] = row['PHQ_Binary']
        if 'PHQ8_Score' in row:
            phq_scores[pid] = row['PHQ8_Score']
    
    # Load features for each participant
    features_by_participant = {}
    valid_participants = []
    
    print(f"Loading {split_path} data...")
    for pid in tqdm(depression_status.keys()):
        feature_path = os.path.join(data_dir, f"{split_path}/{pid}_P/features/{pid}_OpenFace2.1.0_Pose_Gaze_AUs.csv")
        if os.path.exists(feature_path):
            try:
                df = pd.read_csv(feature_path)
                # Filter out non-behavioral features
                behavioral_cols = [col for col in df.columns if is_behavioral_feature(col)]
                features_by_participant[pid] = df[behavioral_cols]
                valid_participants.append(pid)
            except Exception as e:
                print(f"Error loading {pid}: {e}")
    
    # Create labels dictionary for valid participants
    labels = {
        'depression_status': {pid: depression_status[pid] for pid in valid_participants},
        'phq_scores': {pid: phq_scores.get(pid, None) for pid in valid_participants}
    }
    
    return features_by_participant, labels

def analyze_feature_distributions(features_by_participant, labels, feature_name, output_dir):
    """
    Analyze the distribution of a specific feature across depressed and non-depressed groups
    
    Args:
        features_by_participant (dict): Features by participant ID
        labels (dict): Labels including depression status
        feature_name (str): Name of feature to analyze
        output_dir (str): Directory to save output visualizations
    """
    # Check if feature exists in data
    if not any(feature_name in df.columns for df in features_by_participant.values()):
        print(f"Feature {feature_name} not found in the dataset")
        return
    
    # Extract feature values for each group
    depressed_values = []
    non_depressed_values = []
    
    for pid, features_df in features_by_participant.items():
        if feature_name in features_df.columns:
            # Get mean value of feature for this participant (across time)
            feature_mean = features_df[feature_name].mean()
            feature_std = features_df[feature_name].std()
            feature_min = features_df[feature_name].min()
            feature_max = features_df[feature_name].max()
            
            # Create feature summary for this participant
            feature_summary = {
                'mean': feature_mean,
                'std': feature_std,
                'min': feature_min,
                'max': feature_max,
                'values': features_df[feature_name].values
            }
            
            # Add to appropriate group
            if labels['depression_status'][pid] == 1:
                depressed_values.append(feature_summary)
            else:
                non_depressed_values.append(feature_summary)
    
    # Statistical comparison of means
    depressed_means = [v['mean'] for v in depressed_values]
    non_depressed_means = [v['mean'] for v in non_depressed_values]
    
    # Check if we have enough data for statistical testing
    if len(depressed_means) > 1 and len(non_depressed_means) > 1:
        # Perform t-test on means
        t_stat, p_value = stats.ttest_ind(depressed_means, non_depressed_means, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(depressed_means)**2 + np.std(non_depressed_means)**2) / 2)
        effect_size = abs(np.mean(depressed_means) - np.mean(non_depressed_means)) / pooled_std
        
        print(f"\nFeature: {feature_name}")
        print(f"  Depressed (n={len(depressed_means)}): mean={np.mean(depressed_means):.4f}, std={np.std(depressed_means):.4f}")
        print(f"  Non-depressed (n={len(non_depressed_means)}): mean={np.mean(non_depressed_means):.4f}, std={np.std(non_depressed_means):.4f}")
        print(f"  T-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        
        # Interpretation of effect size
        if effect_size < 0.2:
            effect_interpretation = "negligible"
        elif effect_size < 0.5:
            effect_interpretation = "small"
        elif effect_size < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"  Effect size interpretation: {effect_interpretation}")
    else:
        print(f"\nFeature: {feature_name}")
        print(f"  Not enough data for statistical testing")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Distribution of means
    plt.subplot(2, 2, 1)
    plt.hist([non_depressed_means, depressed_means], bins=15, 
             label=['Non-depressed', 'Depressed'], alpha=0.7)
    plt.xlabel(f'{feature_name} Mean Value')
    plt.ylabel('Participant Count')
    plt.title(f'Distribution of Mean {feature_name} Values')
    plt.legend()
    
    # 2. Box plot comparison
    plt.subplot(2, 2, 2)
    box_data = [non_depressed_means, depressed_means]
    plt.boxplot(box_data, labels=['Non-depressed', 'Depressed'])
    plt.ylabel(f'{feature_name} Mean Value')
    plt.title(f'Box Plot of Mean {feature_name} Values')
    
    # 3. Violin plot for more detailed distribution
    plt.subplot(2, 2, 3)
    violin_data = pd.DataFrame({
        'Group': ['Non-depressed'] * len(non_depressed_means) + ['Depressed'] * len(depressed_means),
        'Value': non_depressed_means + depressed_means
    })
    sns.violinplot(x='Group', y='Value', data=violin_data)
    plt.ylabel(f'{feature_name} Mean Value')
    plt.title(f'Violin Plot of Mean {feature_name} Values')
    
    # 4. Time series plot for a sample of participants
    plt.subplot(2, 2, 4)
    
    # Sample up to 3 participants from each group
    sample_depressed = depressed_values[:min(3, len(depressed_values))]
    sample_non_depressed = non_depressed_values[:min(3, len(non_depressed_values))]
    
    # Plot time series for each sampled participant
    for i, sample in enumerate(sample_non_depressed):
        plt.plot(sample['values'][:100], alpha=0.7, color='blue', 
                label='Non-depressed' if i == 0 else "")
    
    for i, sample in enumerate(sample_depressed):
        plt.plot(sample['values'][:100], alpha=0.7, color='red',
                label='Depressed' if i == 0 else "")
    
    plt.xlabel('Frame Index (first 100 frames)')
    plt.ylabel(f'{feature_name} Value')
    plt.title(f'Time Series of {feature_name} (Sample Participants)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{feature_name}_analysis.png"), dpi=300)
    plt.close()
    
    # Save statistical results to CSV
    results_df = pd.DataFrame({
        'Feature': [feature_name],
        'Depressed_Mean': [np.mean(depressed_means) if depressed_means else np.nan],
        'Depressed_Std': [np.std(depressed_means) if depressed_means else np.nan],
        'Depressed_Count': [len(depressed_means)],
        'Non_Depressed_Mean': [np.mean(non_depressed_means) if non_depressed_means else np.nan],
        'Non_Depressed_Std': [np.std(non_depressed_means) if non_depressed_means else np.nan],
        'Non_Depressed_Count': [len(non_depressed_means)],
        'T_Statistic': [t_stat if len(depressed_means) > 1 and len(non_depressed_means) > 1 else np.nan],
        'P_Value': [p_value if len(depressed_means) > 1 and len(non_depressed_means) > 1 else np.nan],
        'Effect_Size': [effect_size if len(depressed_means) > 1 and len(non_depressed_means) > 1 else np.nan],
        'Effect_Interpretation': [effect_interpretation if len(depressed_means) > 1 and len(non_depressed_means) > 1 else "N/A"]
    })
    
    return results_df

def run_full_feature_analysis():
    """Run analysis on all features across all splits"""
    # Create results directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data from all splits
    all_features = {}
    all_labels = {'depression_status': {}, 'phq_scores': {}}
    
    for split in ['train', 'dev', 'test']:
        print(f"\nLoading {split} split...")
        features, labels = load_participants_data(
            split, 
            DATA_DIR, 
            os.path.join(LABELS_DIR, f"{split}_split.csv")
        )
        
        # Add to master dictionaries
        all_features.update(features)
        all_labels['depression_status'].update(labels['depression_status'])
        all_labels['phq_scores'].update(labels['phq_scores'])
    
    # Count participants
    total_participants = len(all_features)
    depressed_count = sum(1 for status in all_labels['depression_status'].values() if status == 1)
    non_depressed_count = total_participants - depressed_count
    
    print(f"\nTotal participants: {total_participants}")
    print(f"Depressed: {depressed_count} ({depressed_count/total_participants*100:.1f}%)")
    print(f"Non-depressed: {non_depressed_count} ({non_depressed_count/total_participants*100:.1f}%)")
    
    # Get common features across all participants
    common_features = set()
    for pid, df in all_features.items():
        if len(common_features) == 0:
            common_features = set(df.columns)
        else:
            common_features = common_features.intersection(set(df.columns))
    
    common_features = list(common_features)
    print(f"\nFound {len(common_features)} common features across all participants")
    
    # Analyze key features
    key_features = ['pose_Tx', 'pose_Ty', 'pose_Tz']  # Start with the ones showing up in importance
    key_features.extend([f for f in common_features if 'AU' in f and not f.endswith('_c')])
    
    # Output statistical results for all features
    all_results = []
    
    print("\nAnalyzing pose features...")
    for feature in key_features[:3]:  # Analyze the key pose features first
        results = analyze_feature_distributions(all_features, all_labels, feature, OUTPUT_DIR)
        if results is not None:
            all_results.append(results)
    
    # Analyze AU features
    print("\nAnalyzing AU features...")
    au_features = [f for f in key_features if 'AU' in f]
    
    # Sort AU features by their numeric identifier
    au_features.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    for feature in au_features:
        results = analyze_feature_distributions(all_features, all_labels, feature, OUTPUT_DIR)
        if results is not None:
            all_results.append(results)
    
    # Combine all results and save to CSV
    if all_results:
        combined_results = pd.concat(all_results)
        combined_results.to_csv(os.path.join(OUTPUT_DIR, "feature_analysis_results.csv"), index=False)
        print(f"\nSaved combined results to: {os.path.join(OUTPUT_DIR, 'feature_analysis_results.csv')}")
    
    # Create correlation matrix of features with depression status
    print("\nCalculating feature correlations with depression status...")
    
    # Extract mean values for each feature and participant
    feature_means = {}
    for feature in common_features:
        feature_means[feature] = []
    
    depression_status_list = []
    
    for pid, df in all_features.items():
        for feature in common_features:
            if feature in df.columns:
                feature_means[feature].append(df[feature].mean())
        
        depression_status_list.append(all_labels['depression_status'][pid])
    
    # Convert to DataFrame
    corr_data = pd.DataFrame(feature_means)
    corr_data['depression_status'] = depression_status_list
    
    # Calculate correlations with depression status
    correlations = {}
    for feature in common_features:
        correlations[feature] = np.corrcoef(corr_data[feature], corr_data['depression_status'])[0, 1]
    
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Save top correlations
    top_correlations = pd.DataFrame(sorted_correlations[:30], columns=['Feature', 'Correlation'])
    top_correlations.to_csv(os.path.join(OUTPUT_DIR, "top_correlations.csv"), index=False)
    
    # Visualize top correlations
    plt.figure(figsize=(10, 8))
    plt.barh(
        [f"{f} ({c:.3f})" for f, c in sorted_correlations[:15]],
        [abs(c) for f, c in sorted_correlations[:15]],
        color=['red' if c < 0 else 'blue' for f, c in sorted_correlations[:15]]
    )
    plt.xlabel('Absolute Correlation with Depression')
    plt.title('Top 15 Features Correlated with Depression Status')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_correlations.png"), dpi=300)
    plt.close()
    
    print("\nAnalysis complete! Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    run_full_feature_analysis()
