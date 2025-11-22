"""Training loop for Temporal Attention Network."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict
import numpy as np
from tqdm import tqdm

from .callbacks.early_stopping import EarlyStopping
from .callbacks.model_checkpoint import ModelCheckpoint


class TANTrainer:
    """Trainer for TAN model."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device(config.device if hasattr(config, 'device') else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate if hasattr(config, 'learning_rate') else 0.001,
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 1e-4
        )
        self.criterion = nn.BCELoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=5)
        
        # Callbacks
        self.early_stopping = EarlyStopping(patience=10)
        self.checkpoint = ModelCheckpoint('results/models/tan_best.pth')
        
        # History
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            if len(batch) == 3:
                sequences, labels, masks = batch
            else:
                sequences, labels = batch[:2]
            
            sequences = sequences.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def validate_epoch(self) -> Dict:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if len(batch) == 3:
                    sequences, labels, masks = batch
                else:
                    sequences, labels = batch[:2]
                
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        predictions = (all_preds >= 0.5).astype(int)
        f1 = f1_score(all_labels, predictions)
        accuracy = accuracy_score(all_labels, predictions)
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'f1': f1,
            'accuracy': accuracy
        }

    def train(self, num_epochs: int) -> Dict:
        """Train for multiple epochs."""
        print(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_metrics = self.validate_epoch()
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['auc'])
            
            # Checkpointing
            self.checkpoint(self.model, val_metrics, epoch)
            
            # Early stopping
            self.early_stopping(val_metrics['auc'])
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        return self.history
