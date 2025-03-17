#!/bin/bash

# Enable error logging
set -e
echo "Starting application setup..."

# Extract the application code if tar.gz exists
if [ -f "/home/site/wwwroot/output.tar.gz" ]; then
    echo "Extracting application from output.tar.gz..."
    mkdir -p /home/app
    tar -xzf /home/site/wwwroot/output.tar.gz -C /home/app
    echo "Extraction complete. App directory contents:"
    ls -la /home/app
fi

# Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
if [ -f "/home/app/requirements.txt" ]; then
    pip install -r /home/app/requirements.txt
else
    pip install -r requirements.txt
fi

# Use the PORT environment variable provided by Azure or default to 8000
export PORT=${PORT:-8000}
echo "Setting up application on port $PORT"

# Change to the app directory
cd /home/app
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

# Look for the app entry point
if [ -f "app.py" ]; then
    APP_MODULE="app:app"
    echo "Found app.py, using $APP_MODULE as entry point"
elif [ -f "main.py" ]; then
    APP_MODULE="main:app"
    echo "Found main.py, using $APP_MODULE as entry point"
elif [ -f "application.py" ]; then
    APP_MODULE="application:app"
    echo "Found application.py, using $APP_MODULE as entry point"
elif [ -f "api.py" ]; then
    APP_MODULE="api:app"
    echo "Found api.py, using $APP_MODULE as entry point"
else
    # List Python files to help find the entry point
    echo "Looking for Python files:"
    find . -name "*.py" -type f | sort
    # Default to main:app if we can't identify the entry point
    APP_MODULE="main:app"
    echo "Could not identify entry point, defaulting to $APP_MODULE"
fi

# Start the application using Gunicorn with proper settings for Azure
echo "Starting app..."
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app