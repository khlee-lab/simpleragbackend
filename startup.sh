#!/bin/bash

# Make sure the script is executable in Linux environments
chmod +x startup.sh

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the application using Gunicorn
gunicorn --bind=0.0.0.0:8000 main:app
