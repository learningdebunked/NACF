# NeuroAdaptive Checkout Framework (NACF)

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of "Designing Inclusive Retail: Federated Learning and Generative Personas for Neurodivergent-Friendly Checkout Experiences"

## 🎯 Overview

NACF is an AI-driven framework for creating neurodivergent-inclusive e-commerce checkout experiences. It uses:

- **Temporal Attention Networks (TAN)** for detecting cognitive friction from clickstream data
- **Generative LLM Personas** for simulating neurodivergent behavior patterns
- **Federated Learning** for privacy-preserving model adaptation
- **Reinforcement Learning** for optimizing adaptive UX

## 📊 Key Results

- **AUC: 0.87** for cognitive overload detection
- **32.4%** cognitive load reduction for ASD users
- **18.7%** overall abandonment reduction
- **Privacy: ε=2.2** differential privacy guarantee

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nacf.git
cd nacf

# Create conda environment
conda env create -f environment.yml
conda activate nacf

# Or use pip
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from src.models.tan.temporal_attention_network import TAN
from src.models.persona_generator.llm_persona_engine import PersonaGenerator

# Initialize TAN model
model = TAN(input_dim=3, cnn_filters=64, gru_hidden=128, attention_heads=4)

# Generate personas
generator = PersonaGenerator()
asd_personas = generator.batch_generate(num_personas=100, persona_type='ASD')
```

### CLI Commands

```bash
# Train TAN model
nacf-train --model tan --epochs 100 --batch-size 64

# Generate personas
nacf-generate-personas --num-asd 400 --num-adhd 400 --num-nt 200

# Evaluate model
nacf-eval --model-path results/models/tan_best.pth

# Run demo
nacf-demo --demo-type persona
```

## 📁 Project Structure

```
nacf/
├── src/
│   ├── config/              # Configuration files
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model architectures
│   │   ├── tan/            # Temporal Attention Network
│   │   ├── persona_generator/  # Persona generation
│   │   ├── federated/      # Federated learning
│   │   └── rl/             # Reinforcement learning
│   ├── training/            # Training loops
│   ├── evaluation/          # Evaluation metrics
│   ├── visualization/       # Plotting utilities
│   └── cli/                 # Command-line interface
├── experiments/             # Hypothesis testing scripts
├── data/                    # Data directories
├── results/                 # Experiment results
└── tests/                   # Unit and integration tests
```

## 🔬 Experiments

The framework validates 5 key hypotheses:

### H1: Friction Detection
Train TAN to detect neurodivergent friction patterns from clickstream data.

```bash
python experiments/hypothesis_testing/h1_friction_detection/train_tan.py
```

### H2: Persona Validity
Generate and validate synthetic neurodivergent personas.

```bash
python experiments/hypothesis_testing/h2_persona_validity/generate_personas.py
```

### H3: Federated Privacy
Train with federated learning and differential privacy.

```bash
python experiments/hypothesis_testing/h3_federated_privacy/train_federated.py
```

### H4: Adaptive UX
Compare UI variants for cognitive load reduction.

```bash
python experiments/hypothesis_testing/h4_adaptive_ux/compare_ui_variants.py
```

### H5: RL Optimization
Optimize checkout flows with reinforcement learning.

```bash
python experiments/hypothesis_testing/h5_rl_optimization/train_rl_agent.py
```

## 📈 Reproducing Paper Results

Run the full reproduction pipeline:

```bash
bash scripts/reproduce_paper_results.sh
```

This will:
1. Download datasets
2. Preprocess data
3. Train all models
4. Run experiments
5. Generate figures
6. Compile results

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# With coverage
pytest --cov=src --cov-report=html
```

## 📊 Model Architecture

### Temporal Attention Network (TAN)

```
Input Sequence → 1D CNN → GRU → Multi-Head Attention → Pooling → Prediction
```

- **Input**: Clickstream sequences (event_type, time_delta, position)
- **CNN**: Local pattern detection (hesitation bursts, rapid clicks)
- **GRU**: Temporal dependency modeling
- **Attention**: Focus on salient friction points
- **Output**: Cognitive overload probability

### Persona Generation

```
Traits → LLM Prompt → Behavioral Description → Parameter Mapping → Persona
```

- **ASD Traits**: Sensory sensitivity, repetitive behavior, predictability preference
- **ADHD Traits**: Attention variance, impulsivity, hyperactivity
- **Output**: Behavioral parameters for simulation

## 🔒 Privacy

The framework implements differential privacy with:
- **ε = 2.2** privacy budget
- **δ = 1e-5** failure probability
- Gradient clipping and Gaussian noise
- Federated aggregation across 40 clients

## 📝 Citation

```bibtex
@article{nacf2024,
  title={Designing Inclusive Retail: Federated Learning and Generative Personas for Neurodivergent-Friendly Checkout Experiences},
  author={Poreddy, Kapil Kumar Reddy},
  journal={JMIR},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the author.

## 🙏 Acknowledgments

- E-commerce datasets: Retailrocket, UCI Online Retail
- Cognitive datasets: MIT GazeCapture, DEAP, ASCERTAIN
- Neurodivergent datasets: OpenNeuro ADHD, Kaggle ASD
- Libraries: PyTorch, Flower, Gymnasium, Transformers
