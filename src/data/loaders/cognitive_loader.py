"""Load cognitive datasets (MIT GazeCapture, DEAP, ASCERTAIN)."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class GazeCaptureLoader:
    """Load MIT GazeCapture dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_gaze_data(self) -> pd.DataFrame:
        """Load gaze variance metrics."""
        # Mock loader - generates synthetic data
        return self._generate_synthetic_gaze_data()
    
    def _generate_synthetic_gaze_data(self, n_participants: int = 200) -> pd.DataFrame:
        """Generate synthetic gaze data."""
        np.random.seed(42)
        data = []
        
        for participant_id in range(n_participants):
            data.append({
                'participant_id': participant_id,
                'variance_x': np.random.uniform(0.1, 2.0),
                'variance_y': np.random.uniform(0.1, 2.0),
                'fixation_count': np.random.randint(10, 100),
                'saccade_velocity': np.random.uniform(50, 300)
            })
        
        return pd.DataFrame(data)


class DEAPLoader:
    """Load DEAP dataset (EEG/physiological)."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_deap_data(self) -> pd.DataFrame:
        """Load EEG/physiological data with arousal/valence labels."""
        return self._generate_synthetic_deap_data()
    
    def _generate_synthetic_deap_data(self, n_participants: int = 200) -> pd.DataFrame:
        """Generate synthetic DEAP data."""
        np.random.seed(42)
        data = []
        
        for participant_id in range(n_participants):
            # Arousal and valence on 1-9 scale
            arousal = np.random.uniform(1, 9)
            valence = np.random.uniform(1, 9)
            
            data.append({
                'participant_id': participant_id,
                'arousal': arousal,
                'valence': valence,
                'eeg_alpha': np.random.uniform(0, 1),
                'eeg_beta': np.random.uniform(0, 1),
                'eeg_theta': np.random.uniform(0, 1),
                'heart_rate': np.random.uniform(60, 100)
            })
        
        return pd.DataFrame(data)


class ASCERTAINLoader:
    """Load ASCERTAIN dataset (multimodal affective)."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_ascertain_data(self) -> pd.DataFrame:
        """Load multimodal affective data."""
        return self._generate_synthetic_ascertain_data()
    
    def _generate_synthetic_ascertain_data(self, n_participants: int = 150) -> pd.DataFrame:
        """Generate synthetic ASCERTAIN data."""
        np.random.seed(42)
        data = []
        
        for participant_id in range(n_participants):
            data.append({
                'participant_id': participant_id,
                'stress_level': np.random.uniform(0, 1),
                'cognitive_load': np.random.uniform(0, 1),
                'engagement': np.random.uniform(0, 1)
            })
        
        return pd.DataFrame(data)


class CognitiveLoadMapper:
    """Map cognitive/affective data to cognitive load levels."""
    
    @staticmethod
    def map_deap_arousal_to_load(arousal_score: float) -> int:
        """Map DEAP arousal (1-9) to cognitive load level (0-2)."""
        if arousal_score < 3:
            return 0  # Low
        elif arousal_score < 7:
            return 1  # Medium
        else:
            return 2  # High
    
    @staticmethod
    def map_gaze_variance_to_load(variance: float, quantiles: tuple = (0.33, 0.66)) -> int:
        """Map gaze variance to cognitive load using quantiles."""
        if variance < quantiles[0]:
            return 0
        elif variance < quantiles[1]:
            return 1
        else:
            return 2
    
    @staticmethod
    def combine_multimodal_features(deap_score: float, gaze_variance: float, 
                                   weights: tuple = (0.6, 0.4)) -> float:
        """Combine multiple cognitive indicators."""
        # Normalize both to 0-1
        deap_norm = (deap_score - 1) / 8  # 1-9 scale to 0-1
        gaze_norm = min(gaze_variance / 2.0, 1.0)  # Cap at 2.0
        
        combined = weights[0] * deap_norm + weights[1] * gaze_norm
        return combined
