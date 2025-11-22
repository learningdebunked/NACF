"""CLI for training models."""

import argparse
import torch
from pathlib import Path

from src.models.tan.temporal_attention_network import TAN
from src.config.model_config import TAN_CONFIG
from src.config.training_config import TRAINING_CONFIG


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train NACF models')
    parser.add_argument('--model', type=str, default='tan', 
                       choices=['tan', 'federated', 'rl'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--data-path', type=str, default='data/processed',
                       help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Training {args.model} model...")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.model == 'tan':
        train_tan(args)
    elif args.model == 'federated':
        print("Federated training not yet implemented")
    elif args.model == 'rl':
        print("RL training not yet implemented")


def train_tan(args):
    """Train TAN model."""
    # Initialize model
    model = TAN(
        input_dim=TAN_CONFIG.input_dim,
        cnn_filters=TAN_CONFIG.cnn_filters,
        gru_hidden=TAN_CONFIG.gru_hidden_size,
        attention_heads=TAN_CONFIG.attention_heads,
        dropout=TAN_CONFIG.dropout
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print("Note: Load your data and create dataloaders before training")
    print(f"Model will be saved to {args.output_dir}/tan_model.pth")


if __name__ == '__main__':
    main()
