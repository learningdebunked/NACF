#!/usr/bin/env python3
"""Train TAN model with real UCI Online Retail data."""

import torch
import numpy as np
from pathlib import Path
import sys

# Data loading
from src.data.loaders.ecommerce_loader import UCIRetailLoader
from src.data.preprocessing.sequence_builder import SequenceBuilder
from src.data.datasets.clickstream_dataset import (
    ClickstreamDataset, get_train_loader, get_val_loader, 
    get_test_loader, split_dataset
)

# Model and training
from src.models.tan.temporal_attention_network import TAN
from src.training.tan_trainer import TANTrainer
from src.config.training_config import TRAINING_CONFIG
from src.evaluation.metrics import evaluate_model


def check_data_exists():
    """Check if real data exists."""
    data_path = Path('data/raw/ecommerce/uci_online_retail/Online_Retail.xlsx')
    if not data_path.exists():
        print("❌ Real data not found!")
        print(f"   Expected: {data_path}")
        print("\n📥 Download real data first:")
        print("   python3 download_uci_retail.py")
        return False
    return True


def main():
    print("=" * 70)
    print("Training TAN with Real UCI Online Retail Data")
    print("=" * 70)
    
    # Check if data exists
    if not check_data_exists():
        sys.exit(1)
    
    # 1. Load real data
    print("\n[1/6] Loading UCI Online Retail dataset...")
    try:
        loader = UCIRetailLoader('data/raw/ecommerce/uci_online_retail')
        df = loader.load_data()
        
        print(f"✓ Loaded {len(df):,} transactions (real data)")
        
        sessions_df = loader.parse_and_group(df)
        print(f"✓ Created {len(sessions_df):,} sessions")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # 2. Build sequences
    print("\n[2/6] Building sequences...")
    builder = SequenceBuilder(seq_length=50, stride=10)
    sequences, labels = builder.build_sequences(sessions_df)
    print(f"✓ Built {len(sequences):,} sequences")
    print(f"✓ Sequence shape: {sequences.shape}")
    print(f"✓ Overload rate: {labels.mean():.2%}")
    
    # 3. Create dataset
    print("\n[3/6] Creating PyTorch dataset...")
    dataset = ClickstreamDataset(sequences, labels)
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, val_split=0.2, test_split=0.1, random_seed=42
    )
    print(f"✓ Train: {len(train_dataset):,} samples")
    print(f"✓ Val:   {len(val_dataset):,} samples")
    print(f"✓ Test:  {len(test_dataset):,} samples")
    
    # 4. Create data loaders
    print("\n[4/6] Creating data loaders...")
    train_loader = get_train_loader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_val_loader(val_dataset, batch_size=64)
    test_loader = get_test_loader(test_dataset, batch_size=64)
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    print(f"✓ Test batches:  {len(test_loader)}")
    
    # 5. Train model
    print("\n[5/6] Training TAN model...")
    print(f"Device: {TRAINING_CONFIG.device}")
    
    model = TAN(
        input_dim=sequences.shape[2],
        cnn_filters=64,
        gru_hidden=128,
        attention_heads=4,
        dropout=0.3
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized with {num_params:,} parameters")
    
    # Create output directory
    Path('results/models').mkdir(parents=True, exist_ok=True)
    
    trainer = TANTrainer(model, train_loader, val_loader, TRAINING_CONFIG)
    
    print("\nStarting training (this may take a while)...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        history = trainer.train(num_epochs=100)
        print(f"\n✓ Training complete!")
        print(f"✓ Best validation AUC: {max(history['val_auc']):.4f}")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Partial model saved")
    
    # 6. Evaluate
    print("\n[6/6] Evaluating on test set...")
    
    try:
        checkpoint = torch.load('results/models/tan_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded best model checkpoint")
    except:
        print("⚠️  Using current model (checkpoint not found)")
    
    test_metrics = evaluate_model(
        model, 
        test_loader, 
        device=TRAINING_CONFIG.device
    )
    
    print("\n" + "=" * 70)
    print("Final Test Results")
    print("=" * 70)
    print(f"AUC:               {test_metrics['auc']:.4f}")
    print(f"F1 Score:          {test_metrics['f1']:.4f}")
    print(f"Precision:         {test_metrics['precision']:.4f}")
    print(f"Recall:            {test_metrics['recall']:.4f}")
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Calibration Error: {test_metrics['calibration_error']:.4f}")
    print("=" * 70)
    
    # Save results
    results_file = Path('results/uci_training_results.txt')
    with open(results_file, 'w') as f:
        f.write("TAN Training Results (Real UCI Online Retail Data)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {len(df):,} transactions, {len(sessions_df):,} sessions\n")
        f.write(f"Sequences: {len(sequences):,}\n")
        f.write(f"Model parameters: {num_params:,}\n\n")
        f.write("Test Metrics:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"\n✓ Results saved to {results_file}")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
