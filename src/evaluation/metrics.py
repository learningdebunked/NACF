"""Evaluation metrics for model performance."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, accuracy_score
)
import torch
from typing import Dict


def calculate_auc(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """Calculate AUC score."""
    return roc_auc_score(y_true, y_pred_probs)


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score."""
    return f1_score(y_true, y_pred)


def calculate_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Calculate precision and recall."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall


def calculate_calibration_error(y_true: np.ndarray, y_pred_probs: np.ndarray, 
                                n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_probs >= bin_lower) & (y_pred_probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def evaluate_model(model, test_loader, device: str = 'cpu') -> Dict:
    """Comprehensive model evaluation."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                sequences, labels, masks = batch
            else:
                sequences, labels = batch[:2]
            
            sequences = sequences.to(device)
            labels = labels.numpy()
            
            outputs = model(sequences)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    metrics = {
        'auc': calculate_auc(all_labels, all_probs),
        'f1': calculate_f1(all_labels, all_preds),
        'accuracy': accuracy_score(all_labels, all_preds),
        'calibration_error': calculate_calibration_error(all_labels, all_probs)
    }
    
    precision, recall = calculate_precision_recall(all_labels, all_preds)
    metrics['precision'] = precision
    metrics['recall'] = recall
    
    return metrics
