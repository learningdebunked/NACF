#!/bin/bash
# Download all available datasets

set -e

echo "========================================="
echo "NACF Dataset Download Script"
echo "========================================="

echo ""
echo "This script will download publicly available datasets."
echo "Some datasets require Kaggle API credentials."
echo ""

# Check if Kaggle API is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠ Kaggle API not configured"
    echo ""
    echo "To download Kaggle datasets, set up API credentials:"
    echo "1. Create account at https://www.kaggle.com"
    echo "2. Go to Account → API → Create New API Token"
    echo "3. Place kaggle.json at ~/.kaggle/kaggle.json"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Continue without Kaggle datasets? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create directories
echo "Creating data directories..."
mkdir -p data/raw/ecommerce/{retailrocket,uci_online_retail}
mkdir -p data/raw/cognitive/{deap,mit_gazecapture,ascertain}
mkdir -p data/raw/neurodivergent/{openneuro_adhd,kaggle_asd}

# Download using Python script
echo ""
echo "Starting downloads..."
python -m src.cli.download_data --all

echo ""
echo "========================================="
echo "Download Complete!"
echo "========================================="
echo ""
echo "Downloaded datasets are in: data/raw/"
echo ""
echo "Next steps:"
echo "1. Preprocess data: python -m src.data.preprocessing.sequence_builder"
echo "2. Train models: nacf-train --model tan"
echo "3. Run experiments: bash scripts/run_all_experiments.sh"
