"""Model hyperparameters configuration."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class TANConfig:
    """Temporal Attention Network configuration."""
    input_dim: int = 128
    cnn_filters: int = 64
    cnn_kernel_size: int = 3
    gru_hidden_size: int = 128
    gru_num_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.3
    output_dim: int = 1


@dataclass
class PersonaConfig:
    """Persona generation configuration."""
    llm_model: str = "gpt2"
    max_length: int = 512
    temperature: float = 0.8
    asd_traits: List[str] = field(default_factory=lambda: [
        "repetitive_scanning",
        "high_sensitivity",
        "predictability_preference"
    ])
    adhd_traits: List[str] = field(default_factory=lambda: [
        "attention_variance",
        "impulsivity",
        "hyperfocus_cycles"
    ])
    num_personas_per_type: int = 400


@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    num_clients: int = 40
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    dp_epsilon: float = 2.2
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0


@dataclass
class RLConfig:
    """Reinforcement learning configuration."""
    state_dim: int = 64
    action_dim: int = 5
    hidden_dim: int = 256
    gamma: float = 0.99
    lr_actor: float = 0.0003
    lr_critic: float = 0.001
    num_episodes: int = 1000
    max_steps_per_episode: int = 50


# Global config instances
TAN_CONFIG = TANConfig()
PERSONA_CONFIG = PersonaConfig()
FEDERATED_CONFIG = FederatedConfig()
RL_CONFIG = RLConfig()
