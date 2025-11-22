# NACF Dataset Guide

This guide explains how to download and use real datasets with NACF.

## 📊 Available Datasets

### E-Commerce Datasets

#### 1. Retailrocket E-Commerce Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- **Size**: ~2.7M events
- **Content**: User clickstream data (views, add-to-cart, transactions)
- **Download**: Automatic (requires Kaggle API)

#### 2. UCI Online Retail Dataset
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Size**: ~540K transactions
- **Content**: Online retail transactions from UK-based store
- **Download**: Automatic (direct download)

### Cognitive/Affective Datasets

#### 3. DEAP Dataset
- **Source**: [DEAP Project](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- **Size**: 32 participants, 40 videos each
- **Content**: EEG, physiological signals, arousal/valence ratings
- **Download**: Manual (requires registration)

#### 4. MIT GazeCapture
- **Source**: [MIT](http://gazecapture.csail.mit.edu/)
- **Size**: 1,450+ participants
- **Content**: Eye tracking data from mobile devices
- **Download**: Manual (requires request)

### Neurodivergent Datasets

#### 5. OpenNeuro ADHD Dataset (ds000030)
- **Source**: [OpenNeuro](https://openneuro.org/datasets/ds000030)
- **Size**: ~200 participants
- **Content**: fMRI data, behavioral measures
- **Download**: Manual (AWS CLI or web)

#### 6. Kaggle ASD Screening Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/faizunnabi/autism-screening)
- **Size**: ~700 records
- **Content**: ASD screening questionnaire responses
- **Download**: Automatic (requires Kaggle API)

---

## 🚀 Quick Start

### Option 1: Automatic Download (Recommended)

```bash
# Download all available datasets
nacf-download-data --all

# Or download specific dataset
nacf-download-data --dataset uci-retail
nacf-download-data --dataset retailrocket
nacf-download-data --dataset kaggle-asd
```

### Option 2: Using Script

```bash
bash scripts/download_all_data.sh
```

### Option 3: Python API

```python
from src.data.downloaders.dataset_downloader import DatasetDownloader

downloader = DatasetDownloader(data_dir='data/raw')

# Download specific datasets
downloader.download_uci_retail()
downloader.download_retailrocket()
downloader.download_kaggle_asd()

# Or download all
downloader.download_all_available()
```

---

## 🔑 Kaggle API Setup

Many datasets require Kaggle API credentials:

### Step 1: Create Kaggle Account
Visit [kaggle.com](https://www.kaggle.com) and create an account.

### Step 2: Get API Token
1. Go to your account settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`

### Step 3: Install Credentials
```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Move credentials file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Step 4: Install Kaggle Package
```bash
pip install kaggle
```

### Step 5: Test
```bash
kaggle datasets list
```

---

## 📥 Manual Download Instructions

### DEAP Dataset

1. Visit: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
2. Fill out the request form
3. Agree to terms of use
4. Download the dataset
5. Extract to: `data/raw/cognitive/deap/`

**Files needed:**
- `data_preprocessed_python/s01.dat` through `s32.dat`
- `participant_ratings.csv`

### MIT GazeCapture

1. Visit: http://gazecapture.csail.mit.edu/
2. Request access to the dataset
3. Download the dataset
4. Extract to: `data/raw/cognitive/mit_gazecapture/`

### OpenNeuro ADHD Dataset

**Option A: AWS CLI**
```bash
# Install AWS CLI
pip install awscli

# Download dataset (large!)
aws s3 sync --no-sign-request \
  s3://openneuro.org/ds000030 \
  data/raw/neurodivergent/openneuro_adhd/
```

**Option B: Web Interface**
1. Visit: https://openneuro.org/datasets/ds000030
2. Click "Download"
3. Extract to: `data/raw/neurodivergent/openneuro_adhd/`

---

## 📁 Expected Directory Structure

After downloading, your data directory should look like:

```
data/raw/
├── ecommerce/
│   ├── retailrocket/
│   │   ├── events.csv
│   │   ├── item_properties_part1.csv
│   │   └── item_properties_part2.csv
│   └── uci_online_retail/
│       └── Online_Retail.xlsx
├── cognitive/
│   ├── deap/
│   │   ├── data_preprocessed_python/
│   │   └── participant_ratings.csv
│   ├── mit_gazecapture/
│   │   └── [gaze data files]
│   └── ascertain/
│       └── [affective data files]
└── neurodivergent/
    ├── openneuro_adhd/
    │   └── [fMRI and behavioral data]
    └── kaggle_asd/
        └── Autism_Data.csv
```

---

## 🔄 Using Downloaded Data

Once datasets are downloaded, update the data loaders:

```python
from src.data.loaders.ecommerce_loader import RetailrocketLoader

# Load real data instead of synthetic
loader = RetailrocketLoader('data/raw/ecommerce/retailrocket')
df = loader.load_events()  # Loads real events.csv

# Parse and group by sessions
sessions = loader.parse_and_group(df, timeout_minutes=30)
```

---

## 📊 Dataset Statistics

| Dataset | Size | Records | Type |
|---------|------|---------|------|
| Retailrocket | ~500 MB | 2.7M events | E-commerce |
| UCI Retail | ~20 MB | 540K transactions | E-commerce |
| DEAP | ~2 GB | 32 participants | Cognitive |
| GazeCapture | ~50 GB | 1,450 participants | Cognitive |
| OpenNeuro ADHD | ~100 GB | 200 participants | Neurodivergent |
| Kaggle ASD | <1 MB | 700 records | Neurodivergent |

---

## ⚠️ Important Notes

### Data Privacy
- All datasets have specific terms of use
- Some require institutional approval
- Do not redistribute without permission
- Cite original sources in publications

### Storage Requirements
- Minimum: ~1 GB (e-commerce + ASD)
- Recommended: ~10 GB (includes DEAP)
- Full: ~150 GB (includes neuroimaging)

### Processing Time
- E-commerce datasets: Minutes
- Cognitive datasets: Hours
- Neuroimaging datasets: Days

---

## 🔧 Troubleshooting

### Kaggle API Issues

**Error: "Could not find kaggle.json"**
```bash
# Check if file exists
ls -la ~/.kaggle/kaggle.json

# If not, download from Kaggle and move it
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Error: "403 Forbidden"**
- Accept dataset terms on Kaggle website first
- Visit dataset page and click "Download"

### Download Failures

**Slow downloads:**
```bash
# Use wget for large files
wget -c <url> -O output_file
```

**Interrupted downloads:**
```bash
# Resume with curl
curl -C - -O <url>
```

### Extraction Issues

**Corrupted archives:**
```bash
# Verify zip file
unzip -t file.zip

# Re-download if corrupted
```

---

## 📚 Dataset Citations

### Retailrocket
```bibtex
@misc{retailrocket2015,
  title={Retailrocket recommender system dataset},
  author={Retailrocket},
  year={2015},
  url={https://www.kaggle.com/retailrocket/ecommerce-dataset}
}
```

### UCI Online Retail
```bibtex
@misc{chen2015,
  title={Online Retail Data Set},
  author={Chen, Daqing},
  year={2015},
  publisher={UCI Machine Learning Repository}
}
```

### DEAP
```bibtex
@article{koelstra2012,
  title={DEAP: A database for emotion analysis using physiological signals},
  author={Koelstra, Sander and others},
  journal={IEEE Transactions on Affective Computing},
  year={2012}
}
```

---

## 🆘 Support

If you encounter issues:

1. **Check dataset availability**: Some datasets may be temporarily unavailable
2. **Verify credentials**: Ensure Kaggle API is properly configured
3. **Check disk space**: Ensure sufficient storage
4. **Review logs**: Check error messages for specific issues

For dataset-specific issues, contact the original data providers.

---

## ✅ Verification

After downloading, verify datasets:

```bash
# Check if files exist
ls -lh data/raw/ecommerce/retailrocket/events.csv
ls -lh data/raw/ecommerce/uci_online_retail/Online_Retail.xlsx

# Test loading
python -c "
from src.data.loaders.ecommerce_loader import RetailrocketLoader
loader = RetailrocketLoader('data/raw/ecommerce/retailrocket')
df = loader.load_events()
print(f'Loaded {len(df)} events')
"
```

---

**Ready to download? Run:**
```bash
nacf-download-data --all
```
