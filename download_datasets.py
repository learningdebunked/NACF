#!/usr/bin/env python3
"""
Direct script to download datasets without installing the package.
Run: python3 download_datasets.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.downloaders.dataset_downloader import DatasetDownloader


def main():
    print("=" * 70)
    print("NACF Dataset Downloader")
    print("=" * 70)
    print("\nThis script will download available datasets.")
    print("Note: Some datasets require Kaggle API credentials.\n")
    
    # Create downloader
    downloader = DatasetDownloader(data_dir='data/raw')
    
    # Download all available
    downloader.download_all_available()
    
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Train with real data: python3 train_with_real_data.py")
    print("2. Or install package: python3 -m pip install -e .")


if __name__ == "__main__":
    main()
