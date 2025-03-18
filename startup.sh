#!/bin/bash

# Make sure this file has LF line endings (not CRLF)
# In Linux, ensure this file is executable with: chmod +x startup.sh

# Enable error logging
set -e
echo "Starting application setup..."

# Azure App Serviceが既に展開済みの場合、そのパスを使用
if [ -n "$ORYX_APP_PATH" ]; then
    APP_HOME="$ORYX_APP_PATH"
    echo "Using Oryx provided app path: $APP_HOME"
elif [ -d "/home/site/wwwroot" ]; then
    APP_HOME="/home/site/wwwroot"
    echo "Using standard Azure App Service location: $APP_HOME"
else
    APP_HOME=$(pwd)
    echo "Using current directory: $APP_HOME"
fi

# Navigate to app directory
cd "$APP_HOME"
echo "Working directory: $(pwd)"
echo "Directory contents:"
ls -la

# Extract output.tar.gz if present
if [ -f "output.tar.gz" ]; then
    echo "Extracting output.tar.gz..."
    tar -xzf output.tar.gz
    echo "Extraction complete."
    echo "Directory contents after extraction:"
    ls -la
fi

# Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found!"
    # Look for requirements.txt in subdirectories
    REQUIREMENTS_PATH=$(find . -name "requirements.txt" -type f | head -n 1)
    if [ ! -z "$REQUIREMENTS_PATH" ]; then
        echo "Found requirements at: $REQUIREMENTS_PATH"
        pip install -r "$REQUIREMENTS_PATH"
    else
        echo "ERROR: No requirements.txt found anywhere!"
        exit 1
    fi
fi

# Handle environment variables
if [ -f ".env" ]; then
    echo "Loading environment from .env file"
    # Note: Pure bash way to load .env without requiring additional tools
    export $(grep -v '^#' .env | xargs)
else
    echo "No .env file found, using system environment variables"
fi

# Check required environment variables - FIX: Use space-separated list instead of array
REQUIRED_VARS="AZURE_STORAGE_CONNECTION_STRING AZURE_SEARCH_ENDPOINT AZURE_SEARCH_KEY AZURE_OPENAI_API_ENDPOINT AZURE_OPENAI_API_KEY AZURE_OPENAI_MODEL_NAME CONTAINER_NAME"
for VAR in $REQUIRED_VARS; do
    if [ -z "$(eval echo \$$VAR)" ]; then
        echo "ERROR: Required environment variable $VAR is not set."
        exit 1
    fi
done

# Use the PORT environment variable provided by Azure or default to 8000
export PORT=${PORT:-8000}
echo "Setting up application on port $PORT"

# Look for the app entry point
APP_MODULE=""
if [ -f "app.py" ]; then
    APP_MODULE="app:app"
    echo "Found app.py, using $APP_MODULE as entry point"
elif [ -f "main.py" ]; then
    # Inspect main.py to confirm app variable name
    APP_VAR=$(grep -o "app = FastAPI" main.py | wc -l)
    if [ "$APP_VAR" -gt 0 ]; then
        APP_MODULE="main:app"
        echo "Found main.py with FastAPI app, using $APP_MODULE as entry point"
    else
        echo "WARNING: main.py found but couldn't confirm FastAPI app variable"
        APP_MODULE="main:app"  # Default assumption
    fi
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

# Make sure Python files are executable
chmod +x *.py 2>/dev/null || true

# Ensure startup.sh is executable
chmod +x *.sh 2>/dev/null || true

# Start the application using Gunicorn with proper settings for Azure
echo "Starting app with module: $APP_MODULE on port $PORT..."
export PYTHONPATH="$APP_HOME:$PYTHONPATH"

# Start with more workers and proper timeout settings
gunicorn --bind=0.0.0.0:$PORT --timeout 600 --access-logfile=- --error-logfile=- -w 4 -k uvicorn.workers.UvicornWorker $APP_MODULE