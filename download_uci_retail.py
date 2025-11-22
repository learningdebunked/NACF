#!/usr/bin/env python3
"""
Download UCI Online Retail dataset (no Kaggle API required).
Run: python3 download_uci_retail.py
"""

import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url, output_path):
    """Download file with progress bar."""
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=output_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Downloaded to {output_path}")


def main():
    print("=" * 70)
    print("Downloading UCI Online Retail Dataset")
    print("=" * 70)
    print("\nThis dataset does NOT require Kaggle API credentials.")
    print("Size: ~20 MB")
    print("Records: 540K transactions\n")
    
    # Output directory
    output_dir = Path('data/raw/ecommerce/uci_online_retail')
    output_path = output_dir / 'Online_Retail.xlsx'
    
    # Check if already exists
    if output_path.exists():
        print(f"✓ Dataset already exists at {output_path}")
        print("Delete it if you want to re-download.")
        return
    
    # Download
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    try:
        download_file(url, output_path)
        
        print("\n" + "=" * 70)
        print("Download Complete!")
        print("=" * 70)
        print(f"\nDataset saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        print("\nNext steps:")
        print("1. Train with this data: python3 train_with_uci_data.py")
        print("2. Or use in your own scripts:")
        print("   from src.data.loaders.ecommerce_loader import UCIRetailLoader")
        print("   loader = UCIRetailLoader('data/raw/ecommerce/uci_online_retail')")
        print("   df = loader.load_data()")
        
    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        print("\nAlternative: Download manually from:")
        print("https://archive.ics.uci.edu/ml/datasets/online+retail")
        print(f"Save to: {output_path}")


if __name__ == "__main__":
    main()
