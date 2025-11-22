"""Training configuration for all models."""

from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    """General training configuration."""
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


@dataclass
class ValidationConfig:
    """Validation configuration."""
    val_split: float = 0.2
    test_split: float = 0.1
    stratify: bool = True
    shuffle: bool = True
    random_seed: int = 42


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_tensorboard: bool = True
    use_wandb: bool = True
    log_interval: int = 10
    save_checkpoint_interval: int = 5
    wandb_project: str = "nacf-experiments"
    wandb_entity: str = "your-username"


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    checkpoint_dir: str = "results/models/"
    save_best_only: bool = True
    metric_to_monitor: str = "val_auc"
    mode: str = "max"


# Global config instances
TRAINING_CONFIG = TrainingConfig()
VALIDATION_CONFIG = ValidationConfig()
LOGGING_CONFIG = LoggingConfig()
CHECKPOINT_CONFIG = CheckpointConfig()
