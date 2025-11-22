# NACF Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Clone and enter directory
cd neuroadaptive-checkoutframework

# 2. Create environment
conda env create -f environment.yml
conda activate nacf

# 3. Install package
pip install -e .
```

## Verify Installation (1 minute)

```bash
# Test imports
python -c "
from src.models.tan.temporal_attention_network import TAN
from src.models.persona_generator.llm_persona_engine import PersonaGenerator
from src.models.rl.a2c_agent import A2CAgent
print('✓ All imports successful!')
"

# Run tests
pytest tests/unit/test_tan_model.py -v
```

## Quick Demo (5 minutes)

### 1. Generate Personas

```python
from src.models.persona_generator.llm_persona_engine import PersonaGenerator

# Initialize generator
generator = PersonaGenerator(model_name="gpt2")

# Generate personas
asd_personas = generator.batch_generate(num_personas=10, persona_type='ASD')
adhd_personas = generator.batch_generate(num_personas=10, persona_type='ADHD')
nt_personas = generator.batch_generate(num_personas=5, persona_type='NT')

print(f"Generated {len(asd_personas)} ASD personas")
print(f"Generated {len(adhd_personas)} ADHD personas")
print(f"Generated {len(nt_personas)} NT personas")

# Inspect a persona
persona = asd_personas[0]
print(f"\nPersona ID: {persona.persona_id}")
print(f"Type: {persona.persona_type}")
print(f"Traits: {persona.traits}")
print(f"Behavioral params: {persona.behavioral_params}")
```

### 2. Train TAN Model

```python
import torch
import numpy as np
from src.models.tan.temporal_attention_network import TAN
from src.data.datasets.clickstream_dataset import ClickstreamDataset, get_train_loader

# Generate synthetic data
n_samples = 1000
sequences = np.random.randn(n_samples, 50, 3)
labels = np.random.randint(0, 2, n_samples)

# Create dataset
dataset = ClickstreamDataset(sequences, labels)
train_loader = get_train_loader(dataset, batch_size=32)

# Initialize model
model = TAN(input_dim=3, cnn_filters=64, gru_hidden=128, attention_heads=4)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Quick training loop (1 epoch)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

for batch in train_loader:
    sequences, labels = batch[:2]
    labels = labels.unsqueeze(1)
    
    optimizer.zero_grad()
    outputs = model(sequences)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Batch loss: {loss.item():.4f}")
    break  # Just one batch for demo

print("✓ Training successful!")
```

### 3. RL Environment

```python
from src.models.rl.environment import CheckoutEnvironment
from src.models.rl.a2c_agent import A2CAgent

# Create environment with personas
env = CheckoutEnvironment(asd_personas + adhd_personas + nt_personas)

# Initialize agent
agent = A2CAgent(state_dim=64, action_dim=5, hidden_dim=256)

# Run one episode
state, _ = env.reset()
total_reward = 0

for step in range(10):  # 10 steps for demo
    action, _ = agent.select_action(state, training=False)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    print(f"Step {step}: Action={action}, Reward={reward:.3f}, Load={info['cognitive_load']:.3f}")
    
    if terminated or truncated:
        break
    
    state = next_state

print(f"\n✓ Episode complete! Total reward: {total_reward:.3f}")
```

## Run Experiments

### H1: Friction Detection

```bash
python experiments/hypothesis_testing/h1_friction_detection/train_tan.py
```

### H2: Persona Generation

```bash
python experiments/hypothesis_testing/h2_persona_validity/generate_personas.py
```

### H5: RL Optimization

```bash
python experiments/hypothesis_testing/h5_rl_optimization/train_rl_agent.py
```

## CLI Usage

```bash
# Generate personas
nacf-generate-personas --num-asd 100 --num-adhd 100 --num-nt 50

# Train model
nacf-train --model tan --epochs 10 --batch-size 32

# Evaluate
nacf-eval --model-path results/models/tan_best.pth
```

## Full Reproduction

```bash
# Run complete pipeline (requires significant compute time)
bash scripts/reproduce_paper_results.sh
```

## Next Steps

1. **Customize configurations**: Edit files in `src/config/`
2. **Add your data**: Place datasets in `data/raw/`
3. **Train full models**: Use training scripts with full epochs
4. **Run all experiments**: Execute all hypothesis testing scripts
5. **Generate visualizations**: Use scripts in `src/visualization/`

## Troubleshooting

### Import Errors
```bash
# Ensure package is installed
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if needed
export CUDA_VISIBLE_DEVICES=""
```

### Memory Issues
```bash
# Reduce batch size in configs
# Edit src/config/training_config.py
# Set batch_size = 16 or 8
```

## Support

- **Documentation**: See README.md
- **Issues**: Open GitHub issue
- **Tests**: Run `pytest tests/ -v`
