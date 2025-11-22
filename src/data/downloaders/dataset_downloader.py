"""Download real datasets from public sources."""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import kaggle


class DatasetDownloader:
    """Download and extract datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, output_path: Path, chunk_size: int = 8192):
        """Download file with progress bar."""
        print(f"Downloading from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=output_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Downloaded to {output_path}")
    
    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract zip file."""
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
    
    def extract_tar(self, tar_path: Path, extract_to: Path):
        """Extract tar file."""
        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
    
    def download_retailrocket(self):
        """
        Download Retailrocket dataset.
        
        Source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
        Note: Requires Kaggle API credentials
        """
        output_dir = self.data_dir / "ecommerce" / "retailrocket"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if (output_dir / "events.csv").exists():
            print("✓ Retailrocket dataset already exists")
            return
        
        try:
            print("Downloading Retailrocket dataset from Kaggle...")
            print("Note: Requires Kaggle API credentials (~/.kaggle/kaggle.json)")
            
            kaggle.api.dataset_download_files(
                'retailrocket/ecommerce-dataset',
                path=output_dir,
                unzip=True
            )
            print(f"✓ Retailrocket dataset downloaded to {output_dir}")
        except Exception as e:
            print(f"✗ Failed to download Retailrocket: {e}")
            print("  Please download manually from:")
            print("  https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset")
    
    def download_uci_retail(self):
        """
        Download UCI Online Retail dataset.
        
        Source: https://archive.ics.uci.edu/ml/datasets/online+retail
        """
        output_dir = self.data_dir / "ecommerce" / "uci_online_retail"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if (output_dir / "Online_Retail.xlsx").exists():
            print("✓ UCI Online Retail dataset already exists")
            return
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        output_path = output_dir / "Online_Retail.xlsx"
        
        try:
            self.download_file(url, output_path)
            print(f"✓ UCI Online Retail downloaded to {output_dir}")
        except Exception as e:
            print(f"✗ Failed to download UCI Online Retail: {e}")
            print("  Please download manually from:")
            print("  https://archive.ics.uci.edu/ml/datasets/online+retail")
    
    def download_deap(self):
        """
        Download DEAP dataset.
        
        Source: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
        Note: Requires registration and agreement to terms
        """
        output_dir = self.data_dir / "cognitive" / "deap"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("DEAP Dataset:")
        print("  This dataset requires registration and agreement to terms.")
        print("  Please visit: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/")
        print("  After registration, download and place files in:")
        print(f"  {output_dir}")
    
    def download_openneuro_adhd(self):
        """
        Download OpenNeuro ADHD dataset.
        
        Source: https://openneuro.org/datasets/ds000030
        Note: Large dataset, requires AWS CLI or direct download
        """
        output_dir = self.data_dir / "neurodivergent" / "openneuro_adhd"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("OpenNeuro ADHD Dataset (ds000030):")
        print("  This is a large neuroimaging dataset.")
        print("  Download options:")
        print("  1. AWS CLI: aws s3 sync --no-sign-request \\")
        print("     s3://openneuro.org/ds000030 ds000030-download/")
        print("  2. Web interface: https://openneuro.org/datasets/ds000030")
        print(f"  Place downloaded files in: {output_dir}")
    
    def download_kaggle_asd(self):
        """
        Download Kaggle ASD screening dataset.
        
        Source: https://www.kaggle.com/datasets/faizunnabi/autism-screening
        Note: Requires Kaggle API credentials
        """
        output_dir = self.data_dir / "neurodivergent" / "kaggle_asd"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if (output_dir / "Autism_Data.csv").exists():
            print("✓ Kaggle ASD dataset already exists")
            return
        
        try:
            print("Downloading Kaggle ASD dataset...")
            print("Note: Requires Kaggle API credentials (~/.kaggle/kaggle.json)")
            
            kaggle.api.dataset_download_files(
                'faizunnabi/autism-screening',
                path=output_dir,
                unzip=True
            )
            print(f"✓ Kaggle ASD dataset downloaded to {output_dir}")
        except Exception as e:
            print(f"✗ Failed to download Kaggle ASD: {e}")
            print("  Please download manually from:")
            print("  https://www.kaggle.com/datasets/faizunnabi/autism-screening")
    
    def download_all_available(self):
        """Download all publicly available datasets."""
        print("=" * 70)
        print("Downloading Available Datasets")
        print("=" * 70)
        
        # Datasets that can be auto-downloaded
        print("\n[1/3] UCI Online Retail")
        print("-" * 70)
        self.download_uci_retail()
        
        print("\n[2/3] Retailrocket (requires Kaggle API)")
        print("-" * 70)
        self.download_retailrocket()
        
        print("\n[3/3] Kaggle ASD (requires Kaggle API)")
        print("-" * 70)
        self.download_kaggle_asd()
        
        # Datasets requiring manual download
        print("\n" + "=" * 70)
        print("Datasets Requiring Manual Download")
        print("=" * 70)
        
        print("\n[1/2] DEAP")
        print("-" * 70)
        self.download_deap()
        
        print("\n[2/2] OpenNeuro ADHD")
        print("-" * 70)
        self.download_openneuro_adhd()
        
        print("\n" + "=" * 70)
        print("Download Summary")
        print("=" * 70)
        print("✓ Automatic downloads completed (where possible)")
        print("⚠ Some datasets require manual download due to:")
        print("  - Registration requirements")
        print("  - Terms of use agreements")
        print("  - Large file sizes")
        print("\nSee instructions above for manual downloads.")


def setup_kaggle_api():
    """Setup instructions for Kaggle API."""
    print("=" * 70)
    print("Kaggle API Setup")
    print("=" * 70)
    print("\nTo download Kaggle datasets, you need API credentials:")
    print("\n1. Create a Kaggle account at https://www.kaggle.com")
    print("2. Go to Account Settings → API → Create New API Token")
    print("3. This downloads kaggle.json")
    print("4. Place it at: ~/.kaggle/kaggle.json")
    print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    print("\nThen install: pip install kaggle")
    print("=" * 70)


if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_all_available()
