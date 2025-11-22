# Training NACF with Real Datasets

Complete guide for training NACF models using real datasets instead of synthetic data.

## 📋 Prerequisites

1. **Download Real Datasets**
   ```bash
   # Download all available datasets
   nacf-download-data --all
   
   # Or download specific ones
   nacf-download-data --dataset uci-retail
   nacf-download-data --dataset retailrocket
   nacf-download-data --dataset kaggle-asd
   ```

2. **Verify Downloads**
   ```bash
   # Check downloaded files
   ls -lh data/raw/ecommerce/retailrocket/events.csv
   ls -lh data/raw/ecommerce/uci_online_retail/Online_Retail.xlsx
   ls -lh data/raw/neurodivergent/kaggle_asd/
   ```

---

## 🎯 Step-by-Step Training Pipeline

### Step 1: Load Real E-Commerce Data

```python
from src.data.loaders.ecommerce_loader import RetailrocketLoader, UCIRetailLoader
import pandas as pd

# Load Retailrocket data
print("Loading Retailrocket dataset...")
retailrocket_loader = RetailrocketLoader('data/raw/ecommerce/retailrocket')
retailrocket_df = retailrocket_loader.load_events()
print(f"Loaded {len(retailrocket_df)} events")

# Parse into sessions
sessions_df = retailrocket_loader.parse_and_group(retailrocket_df, timeout_minutes=30)
print(f"Created {len(sessions_df)} sessions")

# Or load UCI Retail data
print("\nLoading UCI Online Retail dataset...")
uci_loader = UCIRetailLoader('data/raw/ecommerce/uci_online_retail')
uci_df = uci_loader.load_data()
print(f"Loaded {len(uci_df)} transactions")

# Parse into sessions
uci_sessions = uci_loader.parse_and_group(uci_df)
print(f"Created {len(uci_sessions)} sessions")
```

### Step 2: Build Sequences from Real Data

```python
from src.data.preprocessing.sequence_builder import SequenceBuilder
from src.data.preprocessing.feature_engineering import FeatureExtractor
import numpy as np

# Initialize sequence builder
builder = SequenceBuilder(seq_length=50, stride=10)

# Build sequences from real sessions
sequences, labels = builder.build_sequences(sessions_df)
print(f"Built {len(sequences)} sequences")
print(f"Sequence shape: {sequences.shape}")

# Extract features
feature_extractor = FeatureExtractor()
features_list = []

for idx, row in sessions_df.iterrows():
    events = row['event_type']
    timestamps = row['timestamp']
    features = feature_extractor.combine_features(events, timestamps)
    features_list.append(features)

features = np.array(features_list)
print(f"Extracted features shape: {features.shape}")
```

### Step 3: Create Cognitive Load Labels

Since real e-commerce data doesn't have cognitive load labels, we need to create them:

```python
from src.data.cognitive_mapping.load_mapper import CognitiveLoadMapper
from src.data.cognitive_mapping.entropy_calculator import EntropyCalculator

# Calculate interaction entropy as proxy for cognitive load
entropy_calc = EntropyCalculator()
cognitive_loads = []

for idx, row in sessions_df.iterrows():
    events = row['event_type']
    timestamps = row['timestamp']
    
    # Calculate entropy
    entropy = entropy_calc.combined_entropy(events, timestamps)
    
    # Map to cognitive load (0-2 scale)
    if entropy < 3.0:
        load = 0  # Low
    elif entropy < 6.0:
        load = 1  # Medium
    else:
        load = 2  # High
    
    cognitive_loads.append(load)

# Create binary overload labels (1 if load >= 2)
overload_labels = np.array([1 if load >= 2 else 0 for load in cognitive_loads])
print(f"Overload rate: {overload_labels.mean():.2%}")
```

### Step 4: Create PyTorch Dataset

```python
from src.data.datasets.clickstream_dataset import (
    ClickstreamDataset, 
    get_train_loader, 
    get_val_loader,
    get_test_loader,
    split_dataset
)
import torch

# Create dataset with real data
dataset = ClickstreamDataset(
    sequences=sequences,
    labels=overload_labels,
    features=features
)

print(f"Dataset size: {len(dataset)}")

# Split into train/val/test
train_dataset, val_dataset, test_dataset = split_dataset(
    dataset, 
    val_split=0.2, 
    test_split=0.1,
    random_seed=42
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Create data loaders
train_loader = get_train_loader(train_dataset, batch_size=64, shuffle=True)
val_loader = get_val_loader(val_dataset, batch_size=64)
test_loader = get_test_loader(test_dataset, batch_size=64)
```

### Step 5: Train TAN Model

```python
from src.models.tan.temporal_attention_network import TAN
from src.training.tan_trainer import TANTrainer
from src.config.model_config import TAN_CONFIG
from src.config.training_config import TRAINING_CONFIG

# Initialize model
model = TAN(
    input_dim=sequences.shape[2],  # Feature dimension from real data
    cnn_filters=TAN_CONFIG.cnn_filters,
    gru_hidden=TAN_CONFIG.gru_hidden_size,
    attention_heads=TAN_CONFIG.attention_heads,
    dropout=TAN_CONFIG.dropout
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Initialize trainer
trainer = TANTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=TRAINING_CONFIG
)

# Train
print("\nStarting training...")
history = trainer.train(num_epochs=100)

print("\nTraining complete!")
print(f"Best validation AUC: {max(history['val_auc']):.4f}")
```

### Step 6: Evaluate on Test Set

```python
from src.evaluation.metrics import evaluate_model

# Load best model
checkpoint = torch.load('results/models/tan_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
test_metrics = evaluate_model(model, test_loader, device='cuda')

print("\nTest Set Results:")
print(f"AUC: {test_metrics['auc']:.4f}")
print(f"F1: {test_metrics['f1']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Calibration Error: {test_metrics['calibration_error']:.4f}")
```

---

## 🔄 Complete Training Script

Here's a complete script that does everything:

```python
# train_with_real_data.py

import torch
import numpy as np
from pathlib import Path

# Data loading
from src.data.loaders.ecommerce_loader import RetailrocketLoader
from src.data.preprocessing.sequence_builder import SequenceBuilder
from src.data.preprocessing.feature_engineering import FeatureExtractor
from src.data.datasets.clickstream_dataset import (
    ClickstreamDataset, get_train_loader, get_val_loader, 
    get_test_loader, split_dataset
)

# Model and training
from src.models.tan.temporal_attention_network import TAN
from src.training.tan_trainer import TANTrainer
from src.config.model_config import TAN_CONFIG
from src.config.training_config import TRAINING_CONFIG
from src.evaluation.metrics import evaluate_model


def main():
    print("=" * 70)
    print("Training TAN with Real Retailrocket Data")
    print("=" * 70)
    
    # 1. Load real data
    print("\n[1/6] Loading Retailrocket dataset...")
    loader = RetailrocketLoader('data/raw/ecommerce/retailrocket')
    df = loader.load_events()
    sessions_df = loader.parse_and_group(df, timeout_minutes=30)
    print(f"✓ Loaded {len(sessions_df)} sessions")
    
    # 2. Build sequences
    print("\n[2/6] Building sequences...")
    builder = SequenceBuilder(seq_length=50, stride=10)
    sequences, labels = builder.build_sequences(sessions_df)
    print(f"✓ Built {len(sequences)} sequences of shape {sequences.shape}")
    
    # 3. Create dataset
    print("\n[3/6] Creating PyTorch dataset...")
    dataset = ClickstreamDataset(sequences, labels)
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, val_split=0.2, test_split=0.1
    )
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 4. Create data loaders
    print("\n[4/6] Creating data loaders...")
    train_loader = get_train_loader(train_dataset, batch_size=64)
    val_loader = get_val_loader(val_dataset, batch_size=64)
    test_loader = get_test_loader(test_dataset, batch_size=64)
    print(f"✓ Created data loaders")
    
    # 5. Train model
    print("\n[5/6] Training TAN model...")
    model = TAN(
        input_dim=sequences.shape[2],
        cnn_filters=64,
        gru_hidden=128,
        attention_heads=4,
        dropout=0.3
    )
    
    trainer = TANTrainer(model, train_loader, val_loader, TRAINING_CONFIG)
    history = trainer.train(num_epochs=100)
    print(f"✓ Training complete! Best val AUC: {max(history['val_auc']):.4f}")
    
    # 6. Evaluate
    print("\n[6/6] Evaluating on test set...")
    checkpoint = torch.load('results/models/tan_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_loader, device='cuda')
    
    print("\n" + "=" * 70)
    print("Final Test Results")
    print("=" * 70)
    print(f"AUC:        {test_metrics['auc']:.4f}")
    print(f"F1:         {test_metrics['f1']:.4f}")
    print(f"Precision:  {test_metrics['precision']:.4f}")
    print(f"Recall:     {test_metrics['recall']:.4f}")
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Cal Error:  {test_metrics['calibration_error']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

Save this as `train_with_real_data.py` and run:
```bash
python train_with_real_data.py
```

---

## 🧬 Training with Real Neurodivergent Data

### Load Real ASD/ADHD Data

```python
from src.data.loaders.neurodivergent_loader import (
    KaggleASDLoader, 
    OpenNeuroADHDLoader,
    TraitExtractor
)

# Load ASD data
asd_loader = KaggleASDLoader('data/raw/neurodivergent/kaggle_asd')
asd_df = asd_loader.load_asd_data()
print(f"Loaded {len(asd_df)} ASD records")

# Extract traits
trait_extractor = TraitExtractor()
asd_traits = trait_extractor.extract_asd_traits(asd_df)

# Load ADHD data
adhd_loader = OpenNeuroADHDLoader('data/raw/neurodivergent/openneuro_adhd')
adhd_df = adhd_loader.load_adhd_data()
print(f"Loaded {len(adhd_df)} ADHD records")

adhd_traits = trait_extractor.extract_adhd_traits(adhd_df)
```

### Generate Personas from Real Traits

```python
from src.models.persona_generator.llm_persona_engine import PersonaGenerator

generator = PersonaGenerator()

# Generate personas using real trait distributions
real_asd_personas = []
for subject_id, traits in asd_traits.items():
    persona = generator.generate_persona('ASD', traits)
    real_asd_personas.append(persona)

print(f"Generated {len(real_asd_personas)} ASD personas from real data")

real_adhd_personas = []
for subject_id, traits in adhd_traits.items():
    persona = generator.generate_persona('ADHD', traits)
    real_adhd_personas.append(persona)

print(f"Generated {len(real_adhd_personas)} ADHD personas from real data")
```

---

## 🎮 Training RL with Real Personas

```python
from src.models.rl.environment import CheckoutEnvironment
from src.models.rl.a2c_agent import A2CAgent
from src.config.model_config import RL_CONFIG

# Use real personas in RL environment
all_personas = real_asd_personas + real_adhd_personas
env = CheckoutEnvironment(all_personas)

# Initialize agent
agent = A2CAgent(
    state_dim=RL_CONFIG.state_dim,
    action_dim=RL_CONFIG.action_dim,
    hidden_dim=RL_CONFIG.hidden_dim,
    lr_actor=RL_CONFIG.lr_actor,
    lr_critic=RL_CONFIG.lr_critic,
    gamma=RL_CONFIG.gamma
)

# Train
print("Training RL agent with real personas...")
episode_rewards = []

for episode in range(1000):
    total_reward, steps = agent.train_episode(env)
    episode_rewards.append(total_reward)
    
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")

# Save trained agent
agent.save('results/models/rl_agent_real_data.pth')
print("RL agent trained and saved!")
```

---

## 📊 Combining Multiple Real Datasets

```python
# Combine Retailrocket and UCI Retail data
retailrocket_sessions = retailrocket_loader.parse_and_group(retailrocket_df)
uci_sessions = uci_loader.parse_and_group(uci_df)

# Standardize column names
retailrocket_sessions = retailrocket_sessions.rename(columns={
    'session_id': 'session_id',
    'event_type': 'events',
    'timestamp': 'timestamps'
})

uci_sessions = uci_sessions.rename(columns={
    'user_id': 'session_id',
    'items': 'events',
    'timestamp': 'timestamps'
})

# Combine
combined_sessions = pd.concat([retailrocket_sessions, uci_sessions], ignore_index=True)
print(f"Combined dataset: {len(combined_sessions)} sessions")

# Build sequences from combined data
sequences, labels = builder.build_sequences(combined_sessions)
print(f"Built {len(sequences)} sequences from combined data")
```

---

## 🔧 Troubleshooting

### Issue: "File not found"
```bash
# Verify data exists
ls -lh data/raw/ecommerce/retailrocket/events.csv

# If not, download it
nacf-download-data --dataset retailrocket
```

### Issue: "Out of memory"
```python
# Reduce batch size
train_loader = get_train_loader(train_dataset, batch_size=32)  # Instead of 64

# Or reduce sequence length
builder = SequenceBuilder(seq_length=30, stride=10)  # Instead of 50
```

### Issue: "Labels are all the same"
```python
# Check label distribution
print(f"Overload rate: {labels.mean():.2%}")

# Adjust threshold for creating labels
# Make it more balanced
threshold = np.percentile(entropy_values, 70)  # Top 30% are overload
```

---

## ✅ Verification Checklist

- [ ] Real datasets downloaded
- [ ] Data loaded successfully (no synthetic data fallback)
- [ ] Sequences built from real events
- [ ] Labels created (cognitive load proxy)
- [ ] Dataset split into train/val/test
- [ ] Model trained for sufficient epochs
- [ ] Validation metrics improving
- [ ] Test set evaluation performed
- [ ] Results saved to disk

---

## 📈 Expected Results

With real Retailrocket data (~2.7M events):
- Training time: 2-4 hours (GPU)
- Sequences: ~50K-100K
- Target AUC: 0.80-0.87
- Target F1: 0.75-0.81

With combined datasets:
- Training time: 4-8 hours (GPU)
- Sequences: ~100K-200K
- Better generalization
- More robust model

---

## 🎯 Next Steps

1. **Experiment with hyperparameters**
   - Adjust sequence length
   - Try different CNN filters
   - Tune attention heads

2. **Add data augmentation**
   - Temporal jittering
   - Event masking
   - Sequence reversal

3. **Use ensemble methods**
   - Train multiple models
   - Average predictions
   - Boost performance

4. **Fine-tune on specific personas**
   - Train separate models for ASD/ADHD
   - Use transfer learning
   - Personalized predictions

---

**Ready to train? Run:**
```bash
python train_with_real_data.py
```
