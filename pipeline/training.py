"""
Training workflow for neural network models.

Provides single model training with early stopping.
K-fold orchestration and data preparation stay in notebooks.
"""

import torch
import torch.nn as nn
import numpy as np
from pipeline.models import SeqModel, TemporalHuber, prepare_targets


def train_model(X_train, y_train, X_val, y_val, input_dim, horizon, config):
    """
    Train a single model with early stopping.
    
    Args:
        X_train: Training sequences (list of 2D arrays)
        y_train: Training targets (list of 1D arrays)
        X_val: Validation sequences (list of 2D arrays)
        y_val: Validation targets (list of 1D arrays)
        input_dim: Number of input features
        horizon: Prediction horizon (max future frames)
        config: Config object with hyperparameters
    
    Returns:
        model: Trained model (best checkpoint)
        best_loss: Best validation loss achieved
    """
    device = config.DEVICE
    model = SeqModel(input_dim, horizon).to(device)
    
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    
    # Prepare batches
    train_batches = []
    for i in range(0, len(X_train), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by, bm = prepare_targets([y_train[j] for j in range(i, end)], horizon)
        train_batches.append((bx, by, bm))
    
    val_batches = []
    for i in range(0, len(X_val), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets([y_val[j] for j in range(i, end)], horizon)
        val_batches.append((bx, by, bm))
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    # Training loop with early stopping
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        for bx, by, bm in train_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            pred = model(bx)
            loss = criterion(pred, by, bm)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by, bm).item())
        
        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        # Early stopping checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break
    
    # Restore best model state
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_loss

