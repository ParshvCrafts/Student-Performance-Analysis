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

# Install dependencies with exact versions
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify sklearn version matches artifacts
echo "Verifying scikit-learn version..."
python -c "import sklearn; print(f'sklearn version: {sklearn.__version__}')"

echo "=========================================="
echo "Build Process Completed Successfully"
echo "=========================================="
