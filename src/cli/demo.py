"""CLI for running demos."""

import argparse


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Run NACF demo')
    parser.add_argument('--demo-type', type=str, default='persona',
                       choices=['persona', 'detection', 'adaptation'],
                       help='Type of demo to run')
    
    args = parser.parse_args()
    
    print(f"Running {args.demo_type} demo...")
    print("Demo functionality to be implemented")


if __name__ == '__main__':
    main()
