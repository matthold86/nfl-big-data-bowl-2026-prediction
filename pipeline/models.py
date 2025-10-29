"""
Model definitions for neural network architecture.

Contains:
- TemporalHuber: Custom loss function with time decay
- SeqModel: GRU-based sequence model with attention pooling
- prepare_targets: Helper for variable-length target padding
"""

import torch
import torch.nn as nn
import numpy as np


class TemporalHuber(nn.Module):
    """
    Huber loss with temporal decay weighting.
    
    This loss function combines:
    1. Robust Huber loss (less sensitive to outliers than MSE)
    2. Time decay weighting (higher weight for near-term predictions)
    
    Args:
        delta: Huber loss threshold (default: 0.5)
        time_decay: Exponential decay rate for temporal weighting (default: 0.03)
    """
    
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        
        # Huber loss: quadratic for small errors, linear for large errors
        huber = torch.where(abs_err <= self.delta, 0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        # Apply temporal decay weighting
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L)
            huber, mask = huber * weight, mask * weight
        
        # Mask out padded predictions and return averaged loss
        return (huber * mask).sum() / (mask.sum() + 1e-8)


class SeqModel(nn.Module):
    """
    GRU-based sequence model with attention pooling for multi-step prediction.
    
    Architecture:
    1. GRU layers: Capture temporal patterns in input sequences
    2. Attention pooling: Aggregate information across time steps
    3. Prediction head: Output future displacement predictions
    
    Args:
        input_dim: Number of input features
        horizon: Number of future time steps to predict
    """
    
    def __init__(self, input_dim, horizon):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.pool_ln = nn.LayerNorm(128)
        self.pool_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 128))
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, horizon)
        )
    
    def forward(self, x):
        # x: [batch, sequence_length, features]
        h, _ = self.gru(x)  # h: [batch, sequence_length, hidden_dim]
        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)  # Expand learnable query for batch
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))  # Attention pooling
        out = self.head(ctx.squeeze(1))  # Predict future displacements
        return torch.cumsum(out, dim=1)  # Cumulative sum for cumulative displacement


def prepare_targets(batch_axis, max_h):
    """
    Prepare variable-length target arrays for batching.
    
    Pads sequences to max_h length and creates a mask indicating valid predictions.
    
    Args:
        batch_axis: List of 1D arrays with different lengths
        max_h: Maximum horizon (target length)
    
    Returns:
        tensors: Stacked and padded tensor [batch, max_h]
        masks: Binary mask tensor [batch, max_h] where 1 = valid, 0 = padded
    """
    tensors, masks = [], []
    for arr in batch_axis:
        L = len(arr)
        padded = np.pad(arr, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        tensors.append(torch.tensor(padded))
        masks.append(torch.tensor(mask))
    return torch.stack(tensors), torch.stack(masks)

