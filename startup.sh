#!/bin/bash

# Enable error logging
set -e
echo "Starting application setup..."

# Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Use the PORT environment variable provided by Azure or default to 8000
export PORT=${PORT:-8000}
echo "Setting up application on port $PORT"

cd /home/site/wwwroot
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls

# Start the application using Gunicorn with proper settings for Azure
echo "Starting Gunicorn server..."
gunicorn --bind=0.0.0.0:$PORT \
         --workers=4 \
         --timeout=120 \
         --access-logfile=- \
         --error-logfile=- \
         --log-level=info \
         main:app