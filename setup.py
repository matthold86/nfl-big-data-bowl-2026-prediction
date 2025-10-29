from setuptools import setup, find_packages

setup(
    name="nfl_prediction_pipeline",
    version="0.1.0",
    description="Pipeline utilities for NFL Big Data Bowl 2026 prediction",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "polars>=0.19.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
