#!/usr/bin/env python3
"""Test the trained TAN model with real predictions."""

import torch
import numpy as np
from pathlib import Path

from src.models.tan.temporal_attention_network import TAN
from src.data.loaders.ecommerce_loader import UCIRetailLoader
from src.data.preprocessing.sequence_builder import SequenceBuilder
from src.data.preprocessing.feature_engineering import FeatureExtractor


def load_trained_model():
    """Load the trained model from checkpoint."""
    print("=" * 70)
    print("Loading Trained TAN Model")
    print("=" * 70)
    
    checkpoint_path = Path('results/models/tan_best.pth')
    
    if not checkpoint_path.exists():
        print(f"❌ Model checkpoint not found at {checkpoint_path}")
        print("   Please train the model first: python3 train_with_uci_data.py")
        return None
    
    # Initialize model with same architecture
    model = TAN(
        input_dim=3,
        cnn_filters=64,
        gru_hidden=128,
        attention_heads=4,
        dropout=0.3
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"✓ Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"✓ Best metrics: {checkpoint.get('metrics', {})}")
    
    return model


def load_test_data(num_samples=20):
    """Load some test sequences from real data."""
    print("\n" + "=" * 70)
    print("Loading Test Data")
    print("=" * 70)
    
    # Load UCI Retail data
    loader = UCIRetailLoader('data/raw/ecommerce/uci_online_retail')
    df = loader.load_data()
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Parse into sessions
    sessions = loader.parse_and_group(df)
    print(f"✓ Created {len(sessions):,} sessions")
    
    # Build sequences from a subset
    builder = SequenceBuilder(seq_length=50, stride=10)
    sequences, labels = builder.build_sequences(sessions.head(num_samples))
    
    print(f"✓ Built {len(sequences)} test sequences")
    
    return sequences, sessions.head(num_samples)


def analyze_sequence(sequence, session_data, idx):
    """Analyze a single sequence."""
    print(f"\n  Sequence {idx + 1}:")
    print(f"    Customer: {session_data.iloc[0]['user_id']}")
    
    # Extract features
    events = sequence[:, 0]
    time_deltas = sequence[:, 1]
    positions = sequence[:, 2]
    
    # Calculate statistics
    unique_events = len(np.unique(events[events >= 0]))
    avg_time_delta = np.mean(time_deltas[time_deltas > 0]) if np.any(time_deltas > 0) else 0
    max_time_delta = np.max(time_deltas) if len(time_deltas) > 0 else 0
    
    print(f"    Events: {len(events[events >= 0])} items")
    print(f"    Unique items: {unique_events}")
    print(f"    Avg time between events: {avg_time_delta:.2f}s")
    print(f"    Max pause: {max_time_delta:.2f}s")
    
    return {
        'unique_events': unique_events,
        'avg_time_delta': avg_time_delta,
        'max_time_delta': max_time_delta
    }


def make_predictions(model, sequences, sessions):
    """Make predictions on test sequences."""
    print("\n" + "=" * 70)
    print("Making Predictions")
    print("=" * 70)
    
    # Convert to tensor
    sequences_tensor = torch.FloatTensor(sequences)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(sequences_tensor)
        predictions = predictions.squeeze().numpy()
    
    print(f"\n✓ Generated predictions for {len(predictions)} sequences")
    
    # Analyze predictions
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {np.mean(predictions):.4f}")
    print(f"  Std:  {np.std(predictions):.4f}")
    print(f"  Min:  {np.min(predictions):.4f}")
    print(f"  Max:  {np.max(predictions):.4f}")
    
    # Show individual predictions
    print(f"\n{'='*70}")
    print("Individual Predictions (Cognitive Overload Probability)")
    print(f"{'='*70}")
    
    for i in range(min(10, len(predictions))):
        pred = predictions[i]
        sequence = sequences[i]
        
        # Analyze sequence
        stats = analyze_sequence(sequence, sessions, i)
        
        # Show prediction
        risk_level = "HIGH" if pred > 0.7 else "MEDIUM" if pred > 0.3 else "LOW"
        print(f"    Prediction: {pred:.4f} ({risk_level} risk)")
    
    return predictions


def interpret_predictions(predictions, sequences):
    """Interpret what the predictions mean."""
    print("\n" + "=" * 70)
    print("Prediction Interpretation")
    print("=" * 70)
    
    # Classify predictions
    high_risk = np.sum(predictions > 0.7)
    medium_risk = np.sum((predictions > 0.3) & (predictions <= 0.7))
    low_risk = np.sum(predictions <= 0.3)
    
    print(f"\nRisk Distribution:")
    print(f"  🔴 High Risk (>0.7):    {high_risk:3d} sequences ({high_risk/len(predictions)*100:.1f}%)")
    print(f"  🟡 Medium Risk (0.3-0.7): {medium_risk:3d} sequences ({medium_risk/len(predictions)*100:.1f}%)")
    print(f"  🟢 Low Risk (<0.3):     {low_risk:3d} sequences ({low_risk/len(predictions)*100:.1f}%)")
    
    print(f"\nWhat This Means:")
    print(f"  • High risk sequences may indicate cognitive overload")
    print(f"  • These customers might benefit from simplified UI")
    print(f"  • Medium risk customers need moderate assistance")
    print(f"  • Low risk customers can handle standard checkout")
    
    # Find most concerning sequences
    if high_risk > 0:
        print(f"\n⚠️  Most Concerning Sequences:")
        top_indices = np.argsort(predictions)[-5:][::-1]
        for idx in top_indices:
            if predictions[idx] > 0.5:
                print(f"    Sequence {idx+1}: {predictions[idx]:.4f} - Needs attention")


def save_predictions(predictions, output_path='results/predictions.txt'):
    """Save predictions to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("TAN Model Predictions\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total sequences: {len(predictions)}\n")
        f.write(f"Mean prediction: {np.mean(predictions):.4f}\n")
        f.write(f"Std prediction: {np.std(predictions):.4f}\n\n")
        f.write("Individual Predictions:\n")
        for i, pred in enumerate(predictions):
            f.write(f"  Sequence {i+1}: {pred:.4f}\n")
    
    print(f"\n✓ Predictions saved to {output_path}")


def main():
    """Main testing function."""
    print("\n" + "=" * 70)
    print("TAN Model Testing - Real Predictions")
    print("=" * 70)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Load test data
    sequences, sessions = load_test_data(num_samples=20)
    
    if len(sequences) == 0:
        print("❌ No sequences to test")
        return
    
    # Make predictions
    predictions = make_predictions(model, sequences, sessions)
    
    # Interpret results
    interpret_predictions(predictions, sequences)
    
    # Save predictions
    save_predictions(predictions)
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print("\n✅ Your model successfully made predictions on real data!")
    print("\nNext steps:")
    print("  1. Review predictions in results/predictions.txt")
    print("  2. Use model for real-time predictions")
    print("  3. Integrate with adaptive UI system")
    print("  4. Generate personas: python3 -m src.cli.generate_personas")


if __name__ == "__main__":
    main()
