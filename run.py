#!/usr/bin/env python
"""
Run script for the PDF RAG application.

This script provides a convenient way to start the application
with proper logging and error handling.
"""

import os
import sys
import subprocess
import logging
from logger import setup_logger
from config import LOG_LEVEL, LOG_FILE

# Set up logger
logger = setup_logger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import streamlit
        import langchain
        import PyPDF2
        import cohere
        logger.info("All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"Error: Missing dependency - {e}")
        print("Please install all dependencies with: pip install -r requirements.txt")
        return False

def check_cohere_api_key():
    """Check if Cohere API key is set."""
    try:
        from config import COHERE_API_KEY
        if COHERE_API_KEY and COHERE_API_KEY != "":
            logger.info("Cohere API key is set")
            print("Cohere API key is set")

            # Optionally validate the key with a simple API call
            try:
                import cohere
                client = cohere.Client(COHERE_API_KEY)
                # Just get the list of available models to verify the key works
                client.chat(message="test", model="command")
                logger.info("Cohere API key is valid")
                print("Cohere API key is valid")
                return True
            except Exception as e:
                logger.error(f"Error validating Cohere API key: {e}")
                print(f"Warning: Could not validate Cohere API key: {e}")
                print("The application will still start, but you may encounter errors when using Cohere services.")
                return True
        else:
            logger.warning("Cohere API key is not set")
            print("Warning: Cohere API key is not set in config.py")
            print("The application will start, but you need to set a valid API key to use Cohere services.")
            return True
    except Exception as e:
        logger.error(f"Error checking Cohere API key: {e}")
        print(f"Error checking Cohere API key: {e}")
        return False

def run_app():
    """Run the Streamlit application."""
    try:
        logger.info("Starting Streamlit application")
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
        subprocess.run(cmd)
    except Exception as e:
        logger.error(f"Error running Streamlit application: {e}")
        print(f"Error running application: {e}")
        return False
    return True

def main():
    """Main entry point."""
    print(f"Starting PDF RAG Application with Cohere")
    print(f"Log file: {LOG_FILE}")

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check if Cohere API key is set
    check_cohere_api_key()

    # Run the application
    if not run_app():
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
