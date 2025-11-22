"""PyTorch Dataset for clickstream sequences."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Callable


class ClickstreamDataset(Dataset):
    """Dataset for clickstream sequences."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, 
                 features: Optional[np.ndarray] = None, 
                 transform: Optional[Callable] = None):
        """
        Args:
            sequences: Array of shape (N, seq_len, event_dim)
            labels: Array of shape (N,) - binary overload labels
            features: Optional additional features
            transform: Optional data augmentation
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.features = torch.FloatTensor(features) if features is not None else None
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        if self.features is not None:
            return sequence, self.features[idx], label
        else:
            return sequence, label


def collate_fn(batch):
    """Collate function for variable-length sequences."""
    if len(batch[0]) == 3:
        sequences, features, labels = zip(*batch)
        features = torch.stack(features)
    else:
        sequences, labels = zip(*batch)
        features = None
    
    sequences = torch.stack(sequences)
    labels = torch.stack(labels)
    
    # Create attention masks for padding
    masks = (sequences[:, :, 0] != -1).float()
    
    if features is not None:
        return sequences, features, labels, masks
    else:
        return sequences, labels, masks


def get_train_loader(dataset: Dataset, batch_size: int = 32, 
                    shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Create training data loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def get_val_loader(dataset: Dataset, batch_size: int = 32,
                  num_workers: int = 4) -> DataLoader:
    """Create validation data loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def get_test_loader(dataset: Dataset, batch_size: int = 32,
                   num_workers: int = 4) -> DataLoader:
    """Create test data loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def split_dataset(dataset: Dataset, val_split: float = 0.2, 
                 test_split: float = 0.1, random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test."""
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset
