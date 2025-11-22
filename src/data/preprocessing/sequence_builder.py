"""Build temporal event sequences from raw data."""

import numpy as np
import torch
from typing import List, Tuple
import pandas as pd


class SequenceBuilder:
    """Build fixed-length sequences from event data."""
    
    def __init__(self, seq_length: int = 50, stride: int = 10, min_length: int = 5):
        self.seq_length = seq_length
        self.stride = stride
        self.pad_token = -1
        self.min_length = min_length  # Minimum sequence length to keep
    
    def build_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sequences from DataFrame with event lists.
        
        Args:
            df: DataFrame with columns [user_id, event_type/items, timestamp]
        
        Returns:
            sequences: Array of shape (N, seq_length, feature_dim)
            labels: Array of shape (N,)
        """
        sequences = []
        
        for _, row in df.iterrows():
            # Handle different column names (event_type or items)
            if 'event_type' in row:
                events = row['event_type'] if isinstance(row['event_type'], list) else [row['event_type']]
            elif 'items' in row:
                events = row['items'] if isinstance(row['items'], list) else [row['items']]
            else:
                continue
            
            timestamps = row['timestamp'] if isinstance(row['timestamp'], list) else [row['timestamp']]
            
            # Skip if sequence too short
            if len(events) < self.min_length:
                continue
            
            # If shorter than seq_length, pad it
            if len(events) < self.seq_length:
                encoded_seq = self._encode_sequence(events, timestamps)
                sequences.append(encoded_seq)
                continue
            
            # Create sliding windows
            for i in range(0, len(events) - self.seq_length + 1, self.stride):
                seq_events = events[i:i + self.seq_length]
                seq_times = timestamps[i:i + self.seq_length]
                
                # Encode sequence
                encoded_seq = self._encode_sequence(seq_events, seq_times)
                sequences.append(encoded_seq)
        
        if not sequences:
            # Return empty arrays with correct shape
            return np.zeros((0, self.seq_length, 3)), np.zeros(0)
        
        sequences = np.array(sequences)
        
        # Generate synthetic labels (for demonstration)
        labels = self._generate_labels(sequences)
        
        return sequences, labels
    
    def _encode_sequence(self, events: List, timestamps: List) -> np.ndarray:
        """Encode a single sequence with features."""
        encoded = []
        
        for i, (event, timestamp) in enumerate(zip(events, timestamps)):
            # Event type encoding
            event_code = self._encode_event(event)
            
            # Time delta from previous event
            if i == 0:
                time_delta = 0.0
            else:
                try:
                    time_delta = (timestamp - timestamps[i-1]).total_seconds()
                except:
                    time_delta = 0.0
            
            # Session position (0 to 1)
            position = i / len(events)
            
            encoded.append([event_code, time_delta, position])
        
        # Pad if necessary
        while len(encoded) < self.seq_length:
            encoded.append([self.pad_token, 0.0, 0.0])
        
        return np.array(encoded[:self.seq_length])
    
    def _encode_event(self, event) -> int:
        """Encode event type to integer."""
        if isinstance(event, int):
            return event
        
        encoding = {
            'view': 0,
            'click': 0,
            'addtocart': 1,
            'cart': 1,
            'transaction': 2,
            'purchase': 2
        }
        return encoding.get(str(event).lower(), 0)
    
    def _generate_labels(self, sequences: np.ndarray) -> np.ndarray:
        """Generate synthetic overload labels based on sequence complexity."""
        labels = []
        
        for seq in sequences:
            # Calculate complexity metrics
            event_types = seq[:, 0]
            time_deltas = seq[:, 1]
            
            # High variance in timing suggests cognitive load
            time_variance = np.var(time_deltas[time_deltas > 0])
            
            # Many different event types suggests complexity
            unique_events = len(np.unique(event_types[event_types >= 0]))
            
            # Label as overload if high complexity
            overload = 1 if (time_variance > 100 or unique_events > 3) else 0
            labels.append(overload)
        
        return np.array(labels)
    
    def add_special_tokens(self, sequences: np.ndarray) -> np.ndarray:
        """Add START and END tokens to sequences."""
        # Prepend START token (encoded as -2)
        start_token = np.array([[-2, 0.0, 0.0]])
        
        # Append END token (encoded as -3)
        end_token = np.array([[-3, 0.0, 1.0]])
        
        new_sequences = []
        for seq in sequences:
            new_seq = np.vstack([start_token, seq[:-2], end_token])
            new_sequences.append(new_seq)
        
        return np.array(new_sequences)
    
    def sequences_to_tensor(self, sequences: np.ndarray) -> torch.Tensor:
        """Convert numpy sequences to PyTorch tensors."""
        return torch.FloatTensor(sequences)
