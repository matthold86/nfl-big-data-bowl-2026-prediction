"""
Model saving and loading utilities.

Provides functions to save trained models with automatic versioning,
and load them back for inference or further training.
"""

import json
import pickle
import torch
from pathlib import Path
from datetime import datetime


def get_next_version(model_id, models_dir='models'):
    """
    Find the next available version for a model ID.
    
    Args:
        model_id: Base model identifier (e.g., 'nn_baseline')
        models_dir: Directory containing model versions
    
    Returns:
        version: Next available version string (e.g., 'nn_baseline.0')
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up from pipeline/ to project root
    models_path = project_root / models_dir  # Absolute path
    models_path.mkdir(exist_ok=True)
    
    # Find existing versions for this model_id
    existing_versions = []
    for item in models_path.iterdir():
        if item.is_dir() and item.name.startswith(model_id):
            try:
                # Parse version number from directory name
                _, iteration = item.name.split('.')
                existing_versions.append(int(iteration))
            except ValueError:
                continue
    
    # Determine next iteration number
    if existing_versions:
        next_iteration = max(existing_versions) + 1
    else:
        next_iteration = 0
    
    version = f"{model_id}.{next_iteration}"
    return version


def save_model_ensemble(models_x, models_y, scalers, config, metadata, model_id, predictions=None, save_best_only=False):
    """
    Save a trained ensemble to disk with automatic versioning.
    
    Args:
        models_x: List of X-axis models (one per fold)
        models_y: List of Y-axis models (one per fold)
        scalers: List of StandardScaler objects (one per fold)
        config: Config object with hyperparameters
        metadata: Dictionary with training info (losses, features, etc.)
        model_id: Base identifier for the model
        predictions: Optional test set predictions to save
        save_best_only: If True, save only the best fold. If False, save all folds for ensemble.
    
    Returns:
        version_path: Path to the saved model directory
    """
    version = get_next_version(model_id)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    version_path = project_root / 'models' / version  # Absolute path
    version_path.mkdir(exist_ok=True, parents=True)
    
    # Determine which folds to save
    if save_best_only and 'best_fold' in metadata:
        # Save only the best fold
        best_fold = metadata['best_fold']
        torch.save(models_x[best_fold].state_dict(), version_path / "model_x.pt")
        torch.save(models_y[best_fold].state_dict(), version_path / "model_y.pt")
        if len(scalers) > best_fold:
            with open(version_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(scalers[best_fold], f)
    else:
        # Save all folds for ensemble (default, recommended for better predictions)
        for fold_idx, (mx, my) in enumerate(zip(models_x, models_y)):
            torch.save(mx.state_dict(), version_path / f"model_x_fold{fold_idx}.pt")
            torch.save(my.state_dict(), version_path / f"model_y_fold{fold_idx}.pt")
        
        # Save scaler(s)
        with open(version_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(scalers[0] if len(scalers) == 1 else scalers, f)
    
    # Save config
    config_dict = {
        'SEED': config.SEED,
        'N_FOLDS': config.N_FOLDS,
        'BATCH_SIZE': config.BATCH_SIZE,
        'EPOCHS': config.EPOCHS,
        'PATIENCE': config.PATIENCE,
        'LEARNING_RATE': config.LEARNING_RATE,
        'WINDOW_SIZE': config.WINDOW_SIZE,
        'HIDDEN_DIM': config.HIDDEN_DIM,
        'MAX_FUTURE_HORIZON': config.MAX_FUTURE_HORIZON,
        'FIELD_X_MIN': config.FIELD_X_MIN,
        'FIELD_X_MAX': config.FIELD_X_MAX,
        'FIELD_Y_MIN': config.FIELD_Y_MIN,
        'FIELD_Y_MAX': config.FIELD_Y_MAX,
    }
    with open(version_path / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save metadata
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_id'] = model_id
    metadata['version'] = version
    with open(version_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save predictions if provided
    if predictions is not None:
        predictions.to_csv(version_path / 'predictions.csv', index=False)
    
    print(f"\n✓ Model saved to: {version_path}")
    return version_path


def load_model_ensemble(version_path):
    """
    Load a saved model ensemble from disk.
    
    Args:
        version_path: Path to model version directory (e.g., 'models/nn_baseline.0')
    
    Returns:
        dict: Dictionary containing models_x, models_y, scalers, config, metadata
    """
    version_path = Path(version_path)
    
    if not version_path.exists():
        raise ValueError(f"Model path does not exist: {version_path}")
    
    # Load config and metadata
    with open(version_path / 'config.json', 'r') as f:
        config = json.load(f)
    
    with open(version_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load scaler
    with open(version_path / 'scaler.pkl', 'rb') as f:
        scalers = pickle.load(f)
    
    # Find all saved model files
    model_x_files = sorted(version_path.glob('model_x_fold*.pt'))
    model_y_files = sorted(version_path.glob('model_y_fold*.pt'))
    
    # Count how many folds were used (determined by number of saved model files)
    num_folds = len(model_x_files)
    
    if num_folds == 0:
        raise ValueError(f"No model files found in {version_path}")
    
    if len(model_y_files) != num_folds:
        raise ValueError(f"Number of X models ({num_folds}) doesn't match Y models ({len(model_y_files)})")
    
    # Return file paths (actual model loading happens in notebook where device is available)
    return {
        'models_x_files': [str(f) for f in model_x_files],
        'models_y_files': [str(f) for f in model_y_files],
        'scalers': scalers,
        'config': config,
        'metadata': metadata,
        'num_folds': num_folds,
        'version_path': str(version_path)
    }

