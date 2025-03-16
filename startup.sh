#!/bin/bash

# Install dependencies if not already installed
pip install -r requirements.txt

# Start the application with Gunicorn
gunicorn main:app --bind=0.0.0.0:${PORT:-8000} -k uvicorn.workers.UvicornWorker