#!/bin/bash

# Render build script
# This script runs during deployment to set up the environment

set -e  # Exit on error

echo "=========================================="
echo "Starting Build Process"
echo "=========================================="

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if artifacts exist, if not retrain models
if [ ! -f "artifacts/model.pkl" ] || [ ! -f "artifacts/preprocessor.pkl" ]; then
    echo "=========================================="
    echo "Artifacts not found. Retraining models..."
    echo "=========================================="
    python retrain_models.py
else
    echo "=========================================="
    echo "Artifacts found. Skipping retraining."
    echo "To force retrain, delete artifacts folder and redeploy."
    echo "=========================================="
fi

echo "=========================================="
echo "Build Process Completed Successfully"
echo "=========================================="
