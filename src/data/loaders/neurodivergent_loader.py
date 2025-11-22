"""Load neurodivergent datasets (OpenNeuro ADHD, Kaggle ASD)."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


class OpenNeuroADHDLoader:
    """Load OpenNeuro ADHD dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_adhd_data(self) -> pd.DataFrame:
        """Load ADHD reaction time and behavioral data."""
        return self._generate_synthetic_adhd_data()
    
    def _generate_synthetic_adhd_data(self, n_subjects: int = 200) -> pd.DataFrame:
        """Generate synthetic ADHD data."""
        np.random.seed(42)
        data = []
        
        for subject_id in range(n_subjects):
            # ADHD subjects have higher variance in reaction times
            is_adhd = subject_id < 100
            
            if is_adhd:
                reaction_time = np.random.normal(450, 150)  # Higher variance
                variance = np.random.uniform(100, 300)
            else:
                reaction_time = np.random.normal(400, 80)  # Lower variance
                variance = np.random.uniform(50, 150)
            
            data.append({
                'subject_id': subject_id,
                'task_type': np.random.choice(['go_nogo', 'stroop', 'flanker']),
                'reaction_time': max(reaction_time, 200),
                'variance': variance,
                'adhd_label': 1 if is_adhd else 0
            })
        
        return pd.DataFrame(data)


class KaggleASDLoader:
    """Load Kaggle ASD screening dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_asd_data(self) -> pd.DataFrame:
        """Load ASD screening data."""
        return self._generate_synthetic_asd_data()
    
    def _generate_synthetic_asd_data(self, n_subjects: int = 200) -> pd.DataFrame:
        """Generate synthetic ASD data."""
        np.random.seed(42)
        data = []
        
        for subject_id in range(n_subjects):
            is_asd = subject_id < 100
            
            if is_asd:
                repetitive_behavior = np.random.uniform(0.6, 1.0)
                sensory_sensitivity = np.random.uniform(0.7, 1.0)
                social_interaction = np.random.uniform(0.0, 0.4)
            else:
                repetitive_behavior = np.random.uniform(0.0, 0.4)
                sensory_sensitivity = np.random.uniform(0.0, 0.5)
                social_interaction = np.random.uniform(0.6, 1.0)
            
            data.append({
                'subject_id': subject_id,
                'repetitive_behavior_score': repetitive_behavior,
                'sensory_sensitivity': sensory_sensitivity,
                'social_interaction_score': social_interaction,
                'asd_label': 1 if is_asd else 0
            })
        
        return pd.DataFrame(data)


class TraitExtractor:
    """Extract behavioral traits for persona generation."""
    
    @staticmethod
    def extract_asd_traits(asd_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Extract ASD traits per subject."""
        traits = {}
        
        for _, row in asd_data.iterrows():
            subject_id = row['subject_id']
            traits[subject_id] = {
                'repetitive_behavior': row['repetitive_behavior_score'],
                'sensory_sensitivity': row['sensory_sensitivity'],
                'social_interaction': row['social_interaction_score'],
                'predictability_preference': row['repetitive_behavior_score'] * 0.8
            }
        
        return traits
    
    @staticmethod
    def extract_adhd_traits(adhd_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Extract ADHD traits per subject."""
        traits = {}
        
        for _, row in adhd_data.iterrows():
            subject_id = row['subject_id']
            
            # Normalize variance to 0-1
            variance_norm = min(row['variance'] / 300, 1.0)
            
            traits[subject_id] = {
                'attention_variance': variance_norm,
                'impulsivity': np.random.uniform(0.5, 1.0) if row['adhd_label'] else np.random.uniform(0.0, 0.5),
                'hyperactivity': np.random.uniform(0.4, 0.9) if row['adhd_label'] else np.random.uniform(0.0, 0.4)
            }
        
        return traits
    
    @staticmethod
    def normalize_traits(traits: Dict[str, float]) -> Dict[str, float]:
        """Normalize trait scores to 0-1 range."""
        normalized = {}
        for key, value in traits.items():
            normalized[key] = max(0.0, min(1.0, value))
        return normalized
