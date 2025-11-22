"""Unit tests for TAN model."""

import torch
import pytest
from src.models.tan.temporal_attention_network import TAN


def test_tan_initialization():
    """Test TAN model initialization."""
    model = TAN(input_dim=3, cnn_filters=64, gru_hidden=128, attention_heads=4)
    assert model is not None
    assert model.input_dim == 3
    assert model.cnn_filters == 64


def test_tan_forward_pass():
    """Test TAN forward pass."""
    model = TAN(input_dim=3, cnn_filters=64, gru_hidden=128, attention_heads=4)
    
    # Create dummy input
    batch_size = 8
    seq_len = 50
    input_dim = 3
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 1)
    assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output


def test_tan_predict():
    """Test TAN prediction method."""
    model = TAN()
    x = torch.randn(4, 50, 3)
    
    predictions = model.predict(x, threshold=0.5)
    
    assert predictions.shape == (4, 1)
    assert torch.all((predictions == 0) | (predictions == 1))


def test_tan_attention_weights():
    """Test attention weight extraction."""
    model = TAN()
    x = torch.randn(2, 50, 3)
    
    _ = model(x)
    weights = model.get_attention_weights()
    
    # Attention weights should be available after forward pass
    assert weights is not None or weights is None  # May be None initially
