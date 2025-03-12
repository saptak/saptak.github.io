#!/bin/bash
# Simple wrapper script for blog_image_setup.py

# Make the Python script executable
chmod +x blog_image_setup.py

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

# Install required packages using Homebrew (Mac-friendly)
if ! python3 -c "import yaml" &> /dev/null; then
    echo "Installing PyYAML via Homebrew..."
    brew install libyaml
    pip3 install --user pyyaml
fi

if ! python3 -c "import requests" &> /dev/null; then
    echo "Installing requests via pip..."
    pip3 install --user requests
fi

if ! python3 -c "import PIL" &> /dev/null; then
    echo "Installing Pillow via pip..."
    pip3 install --user pillow
fi

# Run the Python script with all arguments passed to this script
python3 blog_image_setup.py "$@"
