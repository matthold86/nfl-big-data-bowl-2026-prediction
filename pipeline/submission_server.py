"""
Submission server wrapper for Kaggle NFL inference API.

HOW IT WORKS:
=============
The Kaggle competition uses a server-based evaluation API. Your model runs as a server
that receives test data in batches and returns predictions. The format is:

1. Define a predict function: predict(test, test_input) -> predictions
2. Create NFLInferenceServer with your predict function
3. Start the server (handles all the networking)

REQUIREMENTS:
- Your predict function must respond within 5 minutes per batch (except first batch has more time)
- Must return Polars or Pandas DataFrame with 'x' and 'y' columns
- Must match the number of predictions to the number of rows in test

TWO MODES:
1. Local testing: Uses a local gateway to simulate the competition environment
2. Competition: Serves predictions to Kaggle's evaluation system
"""

import os
import polars as pl
import pandas as pd


def create_submission_server(predict_fn, gateway_path=None):
    """
    Create and configure a submission server for Kaggle competition.
    
    The server receives test data in batches and calls your predict function.
    It handles all the networking and timing requirements.
    
    Args:
        predict_fn: Function signature: predict(test, test_input) -> DataFrame
                    - test: Test data as Polars DataFrame
                    - test_input: Additional input data as Polars DataFrame  
                    - Returns: Polars or Pandas DataFrame with 'x' and 'y' columns
        gateway_path: Optional tuple/list of paths for local gateway testing
    
    Returns:
        inference_server: NFLInferenceServer instance
    
    Example:
        >>> def my_predict_fn(test, test_input):
        ...     # Load models, transform data, make predictions
        ...     return pl.DataFrame({'x': predictions_x, 'y': predictions_y})
        >>> server = create_submission_server(my_predict_fn)
        >>> server.serve()  # Start the server
    """
    try:
        import kaggle_evaluation.nfl_inference_server
    except ImportError:
        raise ImportError(
            "kaggle_evaluation package not found. This package is only available in Kaggle's competition environment.\n"
            "To test locally, you can mock the predict function. For actual submission, upload your notebook to Kaggle."
        )
    
    server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict_fn)
    
    # Auto-start local gateway if path provided
    if gateway_path is not None:
        if isinstance(gateway_path, str):
            gateway_path = (gateway_path,)  # Convert to tuple if needed
        server.run_local_gateway(gateway_path)
    
    return server


def run_submission_server(predict_fn, gateway_path=None, auto_start=True):
    """
    Create and optionally start the submission server.
    
    Automatically detects competition vs local mode.
    
    Args:
        predict_fn: Function signature: predict(test, test_input) -> DataFrame
        gateway_path: Optional path for local gateway testing
        auto_start: If True, automatically start the server. If False, return server object.
    
    Returns:
        inference_server: NFLInferenceServer instance (started if auto_start=True)
    
    Example:
        >>> # Competition mode (auto-detects)
        >>> run_submission_server(predict_fn)
        
        >>> # Local testing mode
        >>> run_submission_server(predict_fn, gateway_path=('./data/raw/',))
    """
    server = create_submission_server(predict_fn, gateway_path)
    
    if not auto_start:
        return server
    
    # Auto-detect environment and start
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Running in Kaggle competition environment
        print("Starting server in competition mode...")
        server.serve()
    else:
        # Running locally
        if gateway_path:
            print(f"Starting server in local testing mode with gateway: {gateway_path}")
            if isinstance(gateway_path, str):
                gateway_path = (gateway_path,)
            server.run_local_gateway(gateway_path)
        else:
            print("Server created but not started. Set gateway_path to test locally.")
            print("Or call server.serve() manually for production.")
    
    return server

