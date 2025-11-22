"""Integration test for full pipeline."""

import pytest
import torch
import numpy as np
from src.models.tan.temporal_attention_network import TAN
from src.data.datasets.clickstream_dataset import ClickstreamDataset
from src.data.preprocessing.sequence_builder import SequenceBuilder


def test_data_to_model_pipeline():
    """Test full pipeline from data to model."""
    # Create synthetic data
    sequences = np.random.randn(100, 50, 3)
    labels = np.random.randint(0, 2, 100)
    
    # Create dataset
    dataset = ClickstreamDataset(sequences, labels)
    assert len(dataset) == 100
    
    # Create model
    model = TAN(input_dim=3)
    
    # Forward pass
    sample_seq, sample_label = dataset[0]
    output = model(sample_seq.unsqueeze(0))
    
    assert output.shape == (1, 1)
    assert 0 <= output.item() <= 1


def test_persona_generation_pipeline():
    """Test persona generation pipeline."""
    from src.models.persona_generator.llm_persona_engine import PersonaGenerator
    
    generator = PersonaGenerator()
    personas = generator.batch_generate(10, 'ASD')
    
    assert len(personas) == 10
    assert all(p.persona_type == 'ASD' for p in personas)


def test_rl_environment_pipeline():
    """Test RL environment pipeline."""
    from src.models.rl.environment import CheckoutEnvironment
    from src.models.persona_generator.llm_persona_engine import PersonaGenerator
    
    # Generate test personas
    generator = PersonaGenerator()
    personas = generator.batch_generate(5, 'ASD')
    
    # Create environment
    env = CheckoutEnvironment(personas)
    state, _ = env.reset()
    
    assert len(state) == 64
    
    # Take action
    next_state, reward, terminated, truncated, info = env.step(0)
    
    assert len(next_state) == 64
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
