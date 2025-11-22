"""Extract cognitive friction features from sequences."""

import numpy as np
from typing import List, Dict
from collections import Counter
import scipy.stats as stats


class FeatureExtractor:
    """Extract behavioral features from clickstream sequences."""
    
    def extract_t_delta(self, timestamps: List) -> np.ndarray:
        """Compute time differences between consecutive events."""
        if len(timestamps) < 2:
            return np.array([0.0])
        
        deltas = []
        for i in range(1, len(timestamps)):
            try:
                delta = (timestamps[i] - timestamps[i-1]).total_seconds()
                deltas.append(max(delta, 0))
            except:
                deltas.append(0.0)
        
        return np.array(deltas)
    
    def extract_dwell_time(self, timestamps: List, event_types: List) -> Dict[str, float]:
        """Calculate average time spent on each event type."""
        dwell_times = {}
        
        for i in range(len(event_types) - 1):
            event = event_types[i]
            try:
                dwell = (timestamps[i+1] - timestamps[i]).total_seconds()
                if event not in dwell_times:
                    dwell_times[event] = []
                dwell_times[event].append(dwell)
            except:
                pass
        
        # Average dwell time per event type
        avg_dwell = {k: np.mean(v) for k, v in dwell_times.items()}
        return avg_dwell
    
    def extract_interaction_entropy(self, event_sequence: List) -> float:
        """Calculate Shannon entropy of event distribution."""
        if not event_sequence:
            return 0.0
        
        # Count event frequencies
        counts = Counter(event_sequence)
        total = len(event_sequence)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def extract_hesitation_bursts(self, t_deltas: np.ndarray, threshold: float = 5.0) -> int:
        """Count pauses longer than threshold seconds."""
        if len(t_deltas) == 0:
            return 0
        
        hesitations = np.sum(t_deltas > threshold)
        return int(hesitations)
    
    def extract_navigation_loops(self, event_sequence: List, n: int = 3) -> int:
        """Detect repeated back-and-forth patterns using n-grams."""
        if len(event_sequence) < n:
            return 0
        
        # Create n-grams
        ngrams = []
        for i in range(len(event_sequence) - n + 1):
            ngram = tuple(event_sequence[i:i+n])
            ngrams.append(ngram)
        
        # Count repeated n-grams
        ngram_counts = Counter(ngrams)
        loops = sum(1 for count in ngram_counts.values() if count > 1)
        
        return loops
    
    def extract_scroll_oscillation(self, scroll_events: List[float]) -> float:
        """Calculate variance in scroll direction changes."""
        if len(scroll_events) < 2:
            return 0.0
        
        # Calculate direction changes
        directions = []
        for i in range(1, len(scroll_events)):
            direction = 1 if scroll_events[i] > scroll_events[i-1] else -1
            directions.append(direction)
        
        # Variance in directions
        if len(directions) > 0:
            return float(np.var(directions))
        return 0.0
    
    def combine_features(self, sequence: List, timestamps: List) -> np.ndarray:
        """
        Combine all extracted features into single vector.
        
        Returns:
            Feature vector of shape (feature_dim,)
        """
        features = []
        
        # Time-based features
        t_deltas = self.extract_t_delta(timestamps)
        features.extend([
            np.mean(t_deltas) if len(t_deltas) > 0 else 0,
            np.std(t_deltas) if len(t_deltas) > 0 else 0,
            np.max(t_deltas) if len(t_deltas) > 0 else 0,
            np.min(t_deltas) if len(t_deltas) > 0 else 0
        ])
        
        # Entropy
        entropy = self.extract_interaction_entropy(sequence)
        features.append(entropy)
        
        # Hesitation bursts
        hesitations = self.extract_hesitation_bursts(t_deltas)
        features.append(hesitations)
        
        # Navigation loops
        loops = self.extract_navigation_loops(sequence)
        features.append(loops)
        
        # Sequence statistics
        features.extend([
            len(sequence),
            len(set(sequence)),  # Unique events
            len(sequence) / len(set(sequence)) if len(set(sequence)) > 0 else 0  # Repetition ratio
        ])
        
        # Normalize to 0-1 range
        features = np.array(features)
        features = self._normalize_features(features)
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to 0-1 range."""
        # Simple min-max normalization
        min_val = np.min(features)
        max_val = np.max(features)
        
        if max_val - min_val > 0:
            normalized = (features - min_val) / (max_val - min_val)
        else:
            normalized = features
        
        return normalized
