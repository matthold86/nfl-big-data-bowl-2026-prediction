"""
Configuration management with environment awareness.

Provides centralized configuration for training and inference,
with automatic detection of Kaggle vs local environment.
"""

import os
import torch
from pathlib import Path


class Config:
    """
    Centralized configuration for NFL prediction pipeline.
    
    Automatically detects environment (Kaggle vs local) and adjusts paths accordingly.
    """
    
    # Environment-aware data directory
    def __init__(self):
        if os.getenv('KAGGLE_KERNEL_RUN_TYPE'):
            # Running on Kaggle
            self.DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
        else:
            # Running locally
            self.DATA_DIR = Path("./data/raw/train_data")
            
        self.OUTPUT_DIR = Path("./outputs")
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Training hyperparameters
        self.SEED = 42
        self.N_FOLDS = 5
        self.BATCH_SIZE = 256
        self.EPOCHS = 200
        self.PATIENCE = 30
        self.LEARNING_RATE = 1e-3
        
        # Model architecture
        self.WINDOW_SIZE = 10
        self.HIDDEN_DIM = 128
        self.MAX_FUTURE_HORIZON = 94
        
        # Field boundaries
        self.FIELD_X_MIN, self.FIELD_X_MAX = 0.0, 120.0
        self.FIELD_Y_MIN, self.FIELD_Y_MAX = 0.0, 53.3
        
        # Device
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def set_seed(self, seed=42):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def apply_seed(self):
        """Apply the seed to all random generators."""
        self.set_seed(self.SEED)


def set_seed(seed=42):
    """
    Global seed setting function.
    
    Convenience function for setting random seeds.
    Can be used independently of Config class.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

