#!/bin/bash

# Start script for production deployment
# Uses gunicorn for better performance and reliability

echo "Starting Student Performance Predictor..."
gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app
