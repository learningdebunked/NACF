"""Map cognitive/affective data to cognitive load levels."""

import numpy as np
from typing import Dict, List


class CognitiveLoadMapper:
    """Map various cognitive indicators to load levels."""
    
    @staticmethod
    def map_deap_arousal_to_load(arousal_score: float) -> int:
        """
        Map DEAP arousal (1-9) to cognitive load level (0-2).
        
        Args:
            arousal_score: Arousal score on 1-9 scale
        
        Returns:
            0 (Low), 1 (Medium), or 2 (High)
        """
        if arousal_score < 3:
            return 0  # Low
        elif arousal_score < 7:
            return 1  # Medium
        else:
            return 2  # High
    
    @staticmethod
    def map_gaze_variance_to_load(variance: float, 
                                  quantiles: tuple = (0.33, 0.66)) -> int:
        """
        Map gaze variance to cognitive load using quantiles.
        
        Args:
            variance: Gaze variance value
            quantiles: Thresholds for low/medium/high
        
        Returns:
            0 (Low), 1 (Medium), or 2 (High)
        """
        if variance < quantiles[0]:
            return 0
        elif variance < quantiles[1]:
            return 1
        else:
            return 2
    
    @staticmethod
    def combine_multimodal_features(deap_score: float, 
                                   gaze_variance: float,
                                   weights: tuple = (0.6, 0.4)) -> float:
        """
        Combine multiple cognitive indicators.
        
        Args:
            deap_score: DEAP arousal score (1-9)
            gaze_variance: Gaze variance
            weights: Weights for combining (deap, gaze)
        
        Returns:
            Combined cognitive load estimate (0-1)
        """
        # Normalize both to 0-1
        deap_norm = (deap_score - 1) / 8  # 1-9 scale to 0-1
        gaze_norm = min(gaze_variance / 2.0, 1.0)  # Cap at 2.0
        
        combined = weights[0] * deap_norm + weights[1] * gaze_norm
        return combined
    
    @staticmethod
    def entropy_to_load(entropy: float, thresholds: tuple = (3.0, 6.0)) -> int:
        """
        Map interaction entropy to cognitive load.
        
        Args:
            entropy: Interaction entropy value
            thresholds: (low_threshold, high_threshold)
        
        Returns:
            0 (Low), 1 (Medium), or 2 (High)
        """
        if entropy < thresholds[0]:
            return 0
        elif entropy < thresholds[1]:
            return 1
        else:
            return 2
    
    @staticmethod
    def create_labels_from_entropy(entropy_values: np.ndarray,
                                   percentile: float = 70) -> np.ndarray:
        """
        Create binary overload labels from entropy values.
        
        Args:
            entropy_values: Array of entropy values
            percentile: Percentile threshold (top X% are overload)
        
        Returns:
            Binary labels (0 or 1)
        """
        threshold = np.percentile(entropy_values, percentile)
        labels = (entropy_values >= threshold).astype(int)
        return labels
