"""CLI for model evaluation."""

import argparse
import torch
from pathlib import Path

from src.models.tan.temporal_attention_network import TAN
from src.evaluation.metrics import evaluate_model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate NACF models')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data-path', type=str, default='data/processed',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Evaluating model from {args.model_path}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Load your test data and model checkpoint to evaluate")
    print(f"Results will be saved to {args.output_dir}")


if __name__ == '__main__':
    main()
