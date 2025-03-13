"""
Sequence handling utilities for depression severity prediction.

This module provides utilities for processing variable-length sequences,
particularly optimized for xLSTM's capabilities with long sequences.
"""

import torch
import numpy as np

def process_long_sequence(model, sequence, max_chunk_length=5000, overlap=1000, device=None):
    """
    Process a sequence that is too long to fit into memory at once.
    
    This function splits the sequence into overlapping chunks, processes each chunk
    with the model, and combines the results using attention weights.
    
    Args:
        model (nn.Module): The model to use for processing.
        sequence (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
        max_chunk_length (int): Maximum length of each chunk.
        overlap (int): Number of overlapping frames between chunks.
        device (torch.device): Device to use for processing.
        
    Returns:
        tuple: Combined output and final hidden states.
    """
    if device is None:
        device = next(model.parameters()).device
        
    batch_size, seq_length, input_size = sequence.shape
    
    # If sequence is short enough, process directly
    if seq_length <= max_chunk_length:
        return model(sequence.to(device))
    
    # Split sequence into chunks with overlap
    chunks = []
    hidden_states = None
    chunk_start = 0
    
    while chunk_start < seq_length:
        chunk_end = min(chunk_start + max_chunk_length, seq_length)
        chunk = sequence[:, chunk_start:chunk_end, :]
        chunks.append(chunk)
        
        # Next chunk starts after current chunk minus overlap
        chunk_start = chunk_end - overlap
        if chunk_start >= seq_length:
            break
    
    # Process each chunk, maintaining hidden state between chunks
    all_outputs = []
    all_attention_weights = []
    
    with torch.no_grad():  # Use no_grad for efficiency during inference
        for i, chunk in enumerate(chunks):
            # Process chunk
            chunk_output, hidden_states = model(chunk.to(device), hidden_states)
            
            # Store outputs and calculate attention weights for this chunk
            all_outputs.append(chunk_output)
            
            # Simple attention weight calculation based on prediction confidence
            # Higher confidence (lower variance) gets higher weight
            confidence = 1.0 / (torch.var(chunk_output) + 1e-5)
            all_attention_weights.append(confidence.item())
    
    # Normalize attention weights
    weights = np.array(all_attention_weights)
    weights = weights / weights.sum()
    
    # Weighted average of outputs
    final_output = 0
    for i, output in enumerate(all_outputs):
        final_output += output * weights[i]
    
    return final_output, hidden_states


def create_overlapping_windows(sequence, window_size, stride):
    """
    Create overlapping windows from a sequence.
    
    Args:
        sequence (Tensor): Input sequence of shape (seq_length, input_size).
        window_size (int): Size of each window.
        stride (int): Stride between windows.
        
    Returns:
        list: List of windows.
    """
    seq_length = sequence.shape[0]
    windows = []
    
    for i in range(0, seq_length - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
        
    return windows


def apply_exponential_attention(sequence):
    """
    Apply exponential attention to emphasize more recent information.
    
    This is particularly helpful for depression prediction where recent behavior
    may be more relevant than earlier behavior.
    
    Args:
        sequence (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
        
    Returns:
        Tensor: Weighted sequence.
    """
    batch_size, seq_length, input_size = sequence.shape
    
    # Create exponential weights (higher for more recent)
    weights = torch.exp(torch.linspace(0, 2, seq_length))
    weights = weights / weights.sum()
    
    # Reshape weights for broadcasting
    weights = weights.view(1, -1, 1).expand(batch_size, seq_length, input_size)
    weights = weights.to(sequence.device)
    
    # Apply weights
    weighted_sequence = sequence * weights
    
    return weighted_sequence
