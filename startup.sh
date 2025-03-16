#!/bin/bash

# Install dependencies if not already installed
pip install -r requirements.txt

# Start the application with Gunicorn
gunicorn --bind=0.0.0.0 --timeout 600 --workers 4 main:app
