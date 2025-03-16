import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load dotenv, but continue if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.info("python-dotenv package not found. Using default environment variables.")

# Check for required environment variables
def check_required_env_vars(required_env_vars):
    missing_vars = [var for var, value in required_env_vars.items() if not value]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these environment variables before running the application.")
    return missing_vars
