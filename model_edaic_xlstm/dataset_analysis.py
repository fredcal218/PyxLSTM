import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tabulate import tabulate

def analyze_edaic_dataset(base_dir, splits=None):
    """
    Analyze the E-DAIC dataset to calculate average duration and sequence length
    for each split and the overall dataset.
    
    Args:
        base_dir (str): Base directory containing the E-DAIC dataset
        splits (list): List of splits to analyze, defaults to ['train', 'dev', 'test']
    
    Returns:
        dict: Dictionary containing analysis results for each split
    """
    if splits is None:
        splits = ['train', 'dev', 'test']
    
    results = {}
    all_sequences = []
    all_durations = []
    total_participants = 0
    
    # Process each split
    for split in splits:
        print(f"\nAnalyzing {split} split...")
        data_dir = os.path.join(base_dir, 'data_extr', split)
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            print(f"Warning: Directory not found: {data_dir}")
            continue
            
        # Get all participant folders
        participant_folders = [f for f in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, f)) and '_P' in f]
        
        if not participant_folders:
            print(f"No participant folders found in {data_dir}")
            continue
            
        # Initialize statistics collection
        sequences = []
        durations = []
        participant_details = []
        
        # Process each participant
        for folder in participant_folders:
            participant_id = folder.split('_')[0]
            file_path = os.path.join(data_dir, folder, 'features',
                                   f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")
            
            if not os.path.exists(file_path):
                print(f"Warning: Features file not found for participant {participant_id}")
                continue
                
            try:
                # Read only frame and timestamp columns for efficiency
                df = pd.read_csv(file_path, usecols=['frame', 'timestamp'])
                
                # Calculate duration and frame count
                num_frames = len(df)
                if num_frames == 0:
                    print(f"Warning: No frames found for participant {participant_id}")
                    continue
                    
                # Calculate duration in seconds - taking the difference between last and first timestamp
                # Assuming timestamp is in seconds
                if 'timestamp' in df.columns and not df['timestamp'].isna().all():
                    start_time = df['timestamp'].iloc[0]
                    end_time = df['timestamp'].iloc[-1]
                    duration = end_time - start_time
                else:
                    # If no timestamp, estimate duration assuming 30 fps
                    duration = num_frames / 30.0
                
                # Store statistics
                sequences.append(num_frames)
                durations.append(duration)
                participant_details.append({
                    'participant_id': participant_id,
                    'frames': num_frames,
                    'duration': duration,
                    'split': split
                })
                
                print(f"  Participant {participant_id}: {num_frames} frames, {duration:.2f} seconds")
                
            except Exception as e:
                print(f"Error processing participant {participant_id}: {e}")
        
        # Calculate statistics for this split
        if sequences:
            avg_frames = np.mean(sequences)
            std_frames = np.std(sequences)
            min_frames = np.min(sequences)
            max_frames = np.max(sequences)
            
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            min_duration = np.min(durations)
            max_duration = np.max(durations)
            
            # Store results for this split
            results[split] = {
                'num_participants': len(sequences),
                'total_frames': sum(sequences),
                'avg_frames': avg_frames,
                'std_frames': std_frames,
                'min_frames': min_frames,
                'max_frames': max_frames,
                'avg_duration': avg_duration,
                'std_duration': std_duration,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'participant_details': participant_details
            }
            
            # Add to overall statistics
            all_sequences.extend(sequences)
            all_durations.extend(durations)
            total_participants += len(sequences)
            
            # Print summary for this split
            print(f"\n{split.upper()} Split Summary:")
            print(f"  Number of participants: {len(sequences)}")
            print(f"  Average frames per participant: {avg_frames:.2f} ± {std_frames:.2f}")
            print(f"  Average duration per participant: {avg_duration:.2f} ± {std_duration:.2f} seconds")
            print(f"  Frame range: {min_frames} to {max_frames}")
            print(f"  Duration range: {min_duration:.2f} to {max_duration:.2f} seconds")
    
    # Calculate overall statistics
    if all_sequences:
        results['overall'] = {
            'num_participants': total_participants,
            'total_frames': sum(all_sequences),
            'avg_frames': np.mean(all_sequences),
            'std_frames': np.std(all_sequences),
            'min_frames': np.min(all_sequences),
            'max_frames': np.max(all_sequences),
            'avg_duration': np.mean(all_durations),
            'std_duration': np.std(all_durations),
            'min_duration': np.min(all_durations),
            'max_duration': np.max(all_durations)
        }
        
        print("\nOVERALL Dataset Summary:")
        print(f"  Total participants: {total_participants}")
        print(f"  Average frames per participant: {np.mean(all_sequences):.2f} ± {np.std(all_sequences):.2f}")
        print(f"  Average duration per participant: {np.mean(all_durations):.2f} ± {np.std(all_durations):.2f} seconds")
        print(f"  Frame range: {np.min(all_sequences)} to {np.max(all_sequences)}")
        print(f"  Duration range: {np.min(all_durations):.2f} to {np.max(all_durations):.2f} seconds")
        
        # Calculate sequence statistics for model settings
        seq_length = 150  # from your current model settings
        stride = 50  # from your current model settings
        
        print("\nModel Sequence Statistics:")
        print(f"  Current sequence length setting: {seq_length}")
        print(f"  Current stride setting: {stride}")
        
        # Calculate average number of sequences per participant with current settings
        avg_sequences_per_participant = calculate_avg_sequences(all_sequences, seq_length, stride)
        print(f"  Average sequences per participant: {avg_sequences_per_participant:.2f}")
        print(f"  Estimated total sequences in dataset: {avg_sequences_per_participant * total_participants:.2f}")
    
    return results

def calculate_avg_sequences(frame_counts, seq_length, stride):
    """Calculate average number of sequences generated with sliding window"""
    sequences_per_participant = [(max(0, fc - seq_length) // stride) + 1 for fc in frame_counts]
    return np.mean(sequences_per_participant)

def generate_report(results, output_file=None):
    """Generate a formatted report from analysis results"""
    if not results:
        print("No results to report.")
        return
    
    # Create tables for report
    split_table = []
    for split, stats in results.items():
        if split != 'overall':
            split_table.append([
                split,
                stats['num_participants'],
                f"{stats['avg_frames']:.2f} ± {stats['std_frames']:.2f}",
                f"{stats['min_frames']} - {stats['max_frames']}",
                f"{stats['avg_duration']:.2f} ± {stats['std_duration']:.2f}",
                f"{stats['min_duration']:.2f} - {stats['max_duration']:.2f}"
            ])
    
    # Add overall row
    if 'overall' in results:
        overall = results['overall']
        split_table.append([
            'OVERALL',
            overall['num_participants'],
            f"{overall['avg_frames']:.2f} ± {overall['std_frames']:.2f}",
            f"{overall['min_frames']} - {overall['max_frames']}",
            f"{overall['avg_duration']:.2f} ± {overall['std_duration']:.2f}",
            f"{overall['min_duration']:.2f} - {overall['max_duration']:.2f}"
        ])
    
    # Create formatted table
    table = tabulate(
        split_table,
        headers=['Split', 'Participants', 'Avg Frames', 'Frame Range', 'Avg Duration (s)', 'Duration Range (s)'],
        tablefmt='grid'
    )
    
    # Print and save report
    print("\nE-DAIC Dataset Analysis Report")
    print("-----------------------------")
    print(table)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write("E-DAIC Dataset Analysis Report\n")
            f.write("-----------------------------\n\n")
            f.write(table)
            f.write("\n\n")
            
            # Add sequence statistics
            if 'overall' in results:
                seq_length = 150
                stride = 50
                avg_sequences = calculate_avg_sequences(
                    [stats['avg_frames'] for split, stats in results.items() if split != 'overall'], 
                    seq_length, 
                    stride
                )
                f.write(f"\nModel Sequence Statistics (seq_length={seq_length}, stride={stride}):\n")
                f.write(f"Average sequences per participant: {avg_sequences:.2f}\n")
                f.write(f"Estimated total sequences: {avg_sequences * results['overall']['num_participants']:.2f}\n")
        
        print(f"\nReport saved to {output_file}")
    
    return table

def plot_distributions(results, output_dir=None):
    """Plot distributions of frame counts and durations"""
    if not results or 'overall' not in results:
        print("No results to plot.")
        return
    
    # Prepare data
    split_frames = {split: [p['frames'] for p in stats['participant_details']] 
                   for split, stats in results.items() 
                   if split != 'overall' and 'participant_details' in stats}
    
    split_durations = {split: [p['duration'] for p in stats['participant_details']] 
                      for split, stats in results.items() 
                      if split != 'overall' and 'participant_details' in stats}
    
    # Create directory for plots if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot frame distributions
    plt.figure(figsize=(12, 6))
    plt.boxplot([frames for split, frames in split_frames.items()], labels=split_frames.keys())
    plt.title('Distribution of Frame Counts by Split')
    plt.ylabel('Number of Frames')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'frame_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot duration distributions
    plt.figure(figsize=(12, 6))
    plt.boxplot([durations for split, durations in split_durations.items()], labels=split_durations.keys())
    plt.title('Distribution of Recording Durations by Split')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'duration_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Distribution plots created.")

if __name__ == "__main__":
    # Set the base directory for the E-DAIC dataset
    base_dir = 'E-DAIC'  # Change this to the actual path
    
    # Create output directories
    output_dir = os.path.join('model_edaic_xlstm', 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the analysis
    results = analyze_edaic_dataset(base_dir)
    
    # Generate report
    report_file = os.path.join(output_dir, 'dataset_analysis.txt')
    generate_report(results, report_file)
    
    # Generate distribution plots
    plot_distributions(results, output_dir)
    
    print(f"\nAnalysis completed. Results saved to {output_dir}")
