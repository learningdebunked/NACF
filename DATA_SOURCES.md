# NACF Data Sources

Quick reference for all datasets used in NACF.

## 🚀 Quick Download

```bash
# Download all available datasets
nacf-download-data --all

# Or specific datasets
nacf-download-data --dataset uci-retail
nacf-download-data --dataset retailrocket
nacf-download-data --dataset kaggle-asd
```

## 📊 Dataset Summary

| Dataset | Type | Size | Auto-Download | Source |
|---------|------|------|---------------|--------|
| **UCI Online Retail** | E-commerce | 20 MB | ✅ Yes | [UCI](https://archive.ics.uci.edu/ml/datasets/online+retail) |
| **Retailrocket** | E-commerce | 500 MB | ⚠️ Kaggle API | [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) |
| **Kaggle ASD** | Neurodivergent | <1 MB | ⚠️ Kaggle API | [Kaggle](https://www.kaggle.com/datasets/faizunnabi/autism-screening) |
| **DEAP** | Cognitive | 2 GB | ❌ Manual | [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| **MIT GazeCapture** | Cognitive | 50 GB | ❌ Manual | [MIT](http://gazecapture.csail.mit.edu/) |
| **OpenNeuro ADHD** | Neurodivergent | 100 GB | ❌ Manual | [OpenNeuro](https://openneuro.org/datasets/ds000030) |

✅ = Automatic download  
⚠️ = Requires Kaggle API setup  
❌ = Manual download required  

## 🔑 Kaggle API Setup

For datasets marked with ⚠️:

```bash
# 1. Get API token from kaggle.com/account
# 2. Place at ~/.kaggle/kaggle.json
# 3. Set permissions
chmod 600 ~/.kaggle/kaggle.json

# 4. Install Kaggle package
pip install kaggle

# 5. Test
kaggle datasets list
```

Or run:
```bash
nacf-download-data --setup-kaggle
```

## 📖 Full Guide

See [DATASET_GUIDE.md](DATASET_GUIDE.md) for:
- Detailed download instructions
- Manual download procedures
- Dataset descriptions
- Citation information
- Troubleshooting

## 🔄 Using Synthetic Data

If you can't download real datasets, NACF automatically generates synthetic data:

```python
from src.data.loaders.ecommerce_loader import RetailrocketLoader

# Automatically generates synthetic data if real data not found
loader = RetailrocketLoader('data/raw/ecommerce/retailrocket')
df = loader.load_events()  # Returns synthetic data
```

This allows you to:
- Test the framework immediately
- Develop and debug code
- Understand data formats
- Run experiments without waiting for downloads

## 📁 Expected Structure

```
data/raw/
├── ecommerce/
│   ├── retailrocket/events.csv
│   └── uci_online_retail/Online_Retail.xlsx
├── cognitive/
│   └── deap/data_preprocessed_python/
└── neurodivergent/
    ├── openneuro_adhd/
    └── kaggle_asd/Autism_Data.csv
```

## ✅ Verify Downloads

```bash
# Check what's downloaded
ls -lh data/raw/ecommerce/
ls -lh data/raw/neurodivergent/

# Test loading
python -c "
from src.data.loaders.ecommerce_loader import RetailrocketLoader
loader = RetailrocketLoader('data/raw/ecommerce/retailrocket')
df = loader.load_events()
print(f'Loaded {len(df)} events')
"
```
