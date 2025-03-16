#!/bin/bash

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Use the PORT environment variable provided by Azure or default to 8000
PORT=${PORT:-8000}

# Start the application using Gunicorn
gunicorn --bind=0.0.0.0:$PORT main:app