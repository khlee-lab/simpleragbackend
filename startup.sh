#!/bin/bash
# Custom startup script for Azure App Service on Linux for FastAPI

echo "Starting Azure App Service with FastAPI..."

# Navigate to the application root directory (modify as needed)
cd /home/site/wwwroot

# If a requirements.txt exists, install dependencies using pip
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..."
    pip install --no-cache-dir -r requirements.txt
fi

# Launch the FastAPI application using uvicorn
# Assumes your application entrypoint file is 'main.py' with an 'app' instance.
# The $PORT variable is set by Azure App Service. 
echo "Starting FastAPI application..."
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
#gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000}
