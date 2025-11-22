# NACF Installation Guide

## Quick Installation

### Option 1: Using pip (Recommended)

```bash
# Install pip if not available
python3 -m ensurepip --upgrade

# Install the package
python3 -m pip install -e .

# Verify installation
python3 -c "import src; print('✓ NACF installed')"

# Now CLI commands work
nacf-download-data --all
```

### Option 2: Without Installing Package

If you don't want to install the package, you can run scripts directly:

```bash
# Download datasets
python3 download_datasets.py

# Train with real data
python3 train_with_real_data.py

# Generate personas
python3 -m src.cli.generate_personas --num-asd 100

# Run experiments
python3 experiments/hypothesis_testing/h1_friction_detection/train_tan.py
```

### Option 3: Using conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate nacf

# Install package
pip install -e .

# Verify
nacf-download-data --help
```

## Installing Dependencies

### Minimal Installation (Core Only)

```bash
python3 -m pip install torch numpy pandas scikit-learn scipy
```

### Full Installation (All Features)

```bash
python3 -m pip install -r requirements.txt
```

### Individual Dependencies

```bash
# Deep Learning
python3 -m pip install torch torchvision

# Data Science
python3 -m pip install numpy pandas scikit-learn scipy

# Visualization
python3 -m pip install matplotlib seaborn

# NLP
python3 -m pip install transformers

# RL
python3 -m pip install gymnasium

# Federated Learning
python3 -m pip install flwr

# Experiment Tracking
python3 -m pip install tensorboard wandb

# Dataset Download
python3 -m pip install requests kaggle

# Testing
python3 -m pip install pytest
```

## Troubleshooting

### "command not found: pip"

```bash
# Install pip
python3 -m ensurepip --upgrade

# Or use system package manager
# macOS:
brew install python3

# Ubuntu/Debian:
sudo apt-get install python3-pip

# Then try again
python3 -m pip install -e .
```

### "command not found: nacf-download-data"

This means the package isn't installed. Either:

**Option A: Install the package**
```bash
python3 -m pip install -e .
```

**Option B: Use direct scripts**
```bash
python3 download_datasets.py
```

### "ModuleNotFoundError: No module named 'torch'"

```bash
# Install PyTorch
python3 -m pip install torch

# Or install all dependencies
python3 -m pip install -r requirements.txt
```

### "Permission denied"

```bash
# Use --user flag
python3 -m pip install --user -e .

# Or use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv nacf-env

# Activate it
source nacf-env/bin/activate  # macOS/Linux
# Or on Windows: nacf-env\Scripts\activate

# Install package
pip install -e .

# Now CLI commands work
nacf-download-data --all

# Deactivate when done
deactivate
```

## Verification

After installation, verify everything works:

```bash
# Check Python version
python3 --version  # Should be 3.9+

# Check package installation
python3 -c "import src; print('✓ Package found')"

# Check CLI commands
nacf-download-data --help
nacf-train --help
nacf-generate-personas --help

# Run verification script
python3 verify_installation.py
```

## Quick Start After Installation

```bash
# 1. Download datasets
nacf-download-data --dataset uci-retail

# 2. Generate personas
nacf-generate-personas --num-asd 100 --num-adhd 100

# 3. Train model
python3 train_with_real_data.py

# 4. Run experiments
python3 experiments/hypothesis_testing/h1_friction_detection/train_tan.py
```

## Alternative: Docker (Future)

If you prefer Docker:

```bash
# Build image
docker build -t nacf .

# Run container
docker run -it nacf

# Inside container, all commands work
nacf-download-data --all
```

## System Requirements

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for datasets, 5GB for models
- **GPU**: Optional but recommended for training

## Platform-Specific Notes

### macOS
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python3

# Install package
python3 -m pip install -e .
```

### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt-get update

# Install Python and pip
sudo apt-get install python3 python3-pip

# Install package
python3 -m pip install -e .
```

### Windows
```powershell
# Install Python from python.org
# Then in PowerShell:
python -m pip install -e .
```

## Getting Help

If you encounter issues:

1. Check this guide
2. Run `python3 verify_installation.py`
3. Check error messages carefully
4. Ensure Python 3.9+ is installed
5. Try using virtual environment
6. Use direct scripts instead of CLI commands

## Summary

**Easiest method:**
```bash
python3 -m pip install -e .
nacf-download-data --all
```

**Without installation:**
```bash
python3 download_datasets.py
python3 train_with_real_data.py
```

**With virtual environment (best practice):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
nacf-download-data --all
```
