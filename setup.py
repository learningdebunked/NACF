"""Setup configuration for NACF package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

setup(
    name="nacf",
    version="0.1.0",
    author="Kapil Kumar Reddy Poreddy",
    author_email="",
    description="NeuroAdaptive Checkout Framework for neurodivergent-inclusive e-commerce",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nacf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "transformers>=4.30.0",
        "flwr>=1.5.0",
        "gymnasium>=0.28.0",
        "tensorboard>=2.13.0",
        "wandb>=0.15.0",
        "pytest>=7.4.0",
        "jupyter>=1.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "ipywidgets>=8.0.0",
            "openpyxl>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nacf-train=src.cli.train:main",
            "nacf-eval=src.cli.evaluate:main",
            "nacf-demo=src.cli.demo:main",
            "nacf-generate-personas=src.cli.generate_personas:main",
            "nacf-download-data=src.cli.download_data:main",
            "nacf-reproduce=scripts.reproduce_paper_results:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
