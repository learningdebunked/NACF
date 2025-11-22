"""H1: Train TAN for neurodivergent friction detection."""

import torch
import numpy as np
from pathlib import Path

from src.models.tan.temporal_attention_network import TAN
from src.config.model_config import TAN_CONFIG
from src.data.datasets.clickstream_dataset import ClickstreamDataset, get_train_loader, get_val_loader


def main():
    """Train TAN model for H1."""
    print("=" * 60)
    print("H1: Neurodivergent Friction Detection")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("results/h1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data for demonstration
    print("\nGenerating synthetic training data...")
    n_samples = 1000
    seq_len = 50
    input_dim = 3
    
    sequences = np.random.randn(n_samples, seq_len, input_dim)
    labels = np.random.randint(0, 2, n_samples)
    
    # Create dataset
    dataset = ClickstreamDataset(sequences, labels)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = get_train_loader(train_dataset, batch_size=32)
    val_loader = get_val_loader(val_dataset, batch_size=32)
    
    # Initialize model
    print("\nInitializing TAN model...")
    model = TAN(
        input_dim=input_dim,
        cnn_filters=TAN_CONFIG.cnn_filters,
        gru_hidden=TAN_CONFIG.gru_hidden_size,
        attention_heads=TAN_CONFIG.attention_heads,
        dropout=TAN_CONFIG.dropout
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Training would happen here
    print("\nNote: Full training requires running the trainer")
    print("Use: from src.training.tan_trainer import TANTrainer")
    
    # Save model architecture info
    with open(output_dir / "model_info.txt", "w") as f:
        f.write(f"TAN Model\n")
        f.write(f"Parameters: {num_params:,}\n")
        f.write(f"Input dim: {input_dim}\n")
        f.write(f"CNN filters: {TAN_CONFIG.cnn_filters}\n")
        f.write(f"GRU hidden: {TAN_CONFIG.gru_hidden_size}\n")
        f.write(f"Attention heads: {TAN_CONFIG.attention_heads}\n")
    
    print(f"\nResults saved to {output_dir}")
    print("\nTarget metrics:")
    print("- AUC: 0.87")
    print("- F1: 0.81")
    print("- Calibration Error: < 0.05")


if __name__ == "__main__":
    main()
