"""Calculate interaction entropy metrics."""

import numpy as np
from collections import Counter
from typing import List


class EntropyCalculator:
    """Calculate various entropy measures for interaction sequences."""
    
    def shannon_entropy(self, sequence: List) -> float:
        """
        Compute Shannon entropy of event distribution.
        
        Formula: H = -Σ p(i) * log2(p(i))
        
        Args:
            sequence: List of events
        
        Returns:
            Shannon entropy value
        """
        if not sequence:
            return 0.0
        
        # Count event frequencies
        counts = Counter(sequence)
        total = len(sequence)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def temporal_entropy(self, t_deltas: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute entropy of time distribution.
        
        Args:
            t_deltas: Array of time deltas
            n_bins: Number of bins for discretization
        
        Returns:
            Temporal entropy value
        """
        if len(t_deltas) == 0:
            return 0.0
        
        # Discretize time deltas into bins
        hist, _ = np.histogram(t_deltas, bins=n_bins)
        
        # Calculate entropy
        total = np.sum(hist)
        if total == 0:
            return 0.0
        
        probs = hist / total
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

    def transition_entropy(self, sequence: List) -> float:
        """
        Calculate entropy of state transitions.
        
        Args:
            sequence: List of events
        
        Returns:
            Transition entropy value
        """
        if len(sequence) < 2:
            return 0.0
        
        # Build transition counts
        transitions = []
        for i in range(len(sequence) - 1):
            transition = (sequence[i], sequence[i+1])
            transitions.append(transition)
        
        # Calculate entropy
        counts = Counter(transitions)
        total = len(transitions)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def combined_entropy(self, sequence: List, t_deltas: np.ndarray = None,
                        weights: List[float] = [0.5, 0.3, 0.2]) -> float:
        """
        Weighted combination of all entropy measures.
        
        Args:
            sequence: List of events
            t_deltas: Optional time deltas
            weights: Weights for [shannon, temporal, transition]
        
        Returns:
            Combined entropy (0-10 scale)
        """
        # Shannon entropy
        shannon = self.shannon_entropy(sequence)
        
        # Temporal entropy
        if t_deltas is not None and len(t_deltas) > 0:
            temporal = self.temporal_entropy(t_deltas)
        else:
            temporal = 0.0
        
        # Transition entropy
        transition = self.transition_entropy(sequence)
        
        # Weighted combination
        combined = (weights[0] * shannon + 
                   weights[1] * temporal + 
                   weights[2] * transition)
        
        # Scale to 0-10
        combined = combined * 2.0  # Approximate scaling
        
        return combined
