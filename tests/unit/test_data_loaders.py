"""Unit tests for data loaders."""

import pytest
import pandas as pd
from src.data.loaders.ecommerce_loader import RetailrocketLoader, UCIRetailLoader
from src.data.loaders.cognitive_loader import DEAPLoader, GazeCaptureLoader
from src.data.loaders.neurodivergent_loader import OpenNeuroADHDLoader, KaggleASDLoader


def test_retailrocket_loader():
    """Test Retailrocket loader."""
    loader = RetailrocketLoader('data/raw/ecommerce/retailrocket')
    df = loader.load_events()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_uci_retail_loader():
    """Test UCI Retail loader."""
    loader = UCIRetailLoader('data/raw/ecommerce/uci_online_retail')
    df = loader.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_deap_loader():
    """Test DEAP loader."""
    loader = DEAPLoader('data/raw/cognitive/deap')
    df = loader.load_deap_data()
    
    assert isinstance(df, pd.DataFrame)
    assert 'arousal' in df.columns
    assert 'valence' in df.columns


def test_gaze_capture_loader():
    """Test GazeCapture loader."""
    loader = GazeCaptureLoader('data/raw/cognitive/mit_gazecapture')
    df = loader.load_gaze_data()
    
    assert isinstance(df, pd.DataFrame)
    assert 'variance_x' in df.columns


def test_adhd_loader():
    """Test ADHD loader."""
    loader = OpenNeuroADHDLoader('data/raw/neurodivergent/openneuro_adhd')
    df = loader.load_adhd_data()
    
    assert isinstance(df, pd.DataFrame)
    assert 'reaction_time' in df.columns


def test_asd_loader():
    """Test ASD loader."""
    loader = KaggleASDLoader('data/raw/neurodivergent/kaggle_asd')
    df = loader.load_asd_data()
    
    assert isinstance(df, pd.DataFrame)
    assert 'sensory_sensitivity' in df.columns
