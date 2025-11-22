"""Model checkpointing callback."""

import torch
from pathlib import Path


class ModelCheckpoint:
    """Save model checkpoints."""
    
    def __init__(self, filepath: str, monitor: str = 'val_auc', 
                 mode: str = 'max', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, model, current_metrics: dict, epoch: int):
        """Save checkpoint if improved."""
        current_score = current_metrics.get(self.monitor.replace('val_', ''))
        
        if current_score is None:
            return
        
        if self.is_best(current_score):
            self.best_score = current_score
            self.save_checkpoint(model, current_metrics, epoch)
    
    def is_best(self, score: float) -> bool:
        """Check if current score is best."""
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return score > self.best_score
        else:
            return score < self.best_score
    
    def save_checkpoint(self, model, metrics: dict, epoch: int):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, self.filepath)
        print(f"Checkpoint saved to {self.filepath}")
