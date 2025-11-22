"""CLI for downloading datasets."""

import argparse
from src.data.downloaders.dataset_downloader import DatasetDownloader, setup_kaggle_api


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(
        description='Download NACF datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all available datasets
  nacf-download-data --all

  # Download specific dataset
  nacf-download-data --dataset retailrocket

  # Show Kaggle API setup instructions
  nacf-download-data --setup-kaggle

Available datasets:
  - retailrocket: Retailrocket e-commerce dataset (Kaggle)
  - uci-retail: UCI Online Retail dataset (direct download)
  - kaggle-asd: Kaggle ASD screening dataset (Kaggle)
  - deap: DEAP emotion dataset (manual download required)
  - openneuro-adhd: OpenNeuro ADHD dataset (manual download required)
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available datasets'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['retailrocket', 'uci-retail', 'kaggle-asd', 'deap', 'openneuro-adhd'],
        help='Download specific dataset'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Data directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--setup-kaggle',
        action='store_true',
        help='Show Kaggle API setup instructions'
    )
    
    args = parser.parse_args()
    
    if args.setup_kaggle:
        setup_kaggle_api()
        return
    
    downloader = DatasetDownloader(data_dir=args.data_dir)
    
    if args.all:
        downloader.download_all_available()
    elif args.dataset:
        dataset_map = {
            'retailrocket': downloader.download_retailrocket,
            'uci-retail': downloader.download_uci_retail,
            'kaggle-asd': downloader.download_kaggle_asd,
            'deap': downloader.download_deap,
            'openneuro-adhd': downloader.download_openneuro_adhd
        }
        
        print(f"Downloading {args.dataset}...")
        dataset_map[args.dataset]()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
