#!/bin/bash
# Setup blog images using a Python virtual environment (recommended for macOS)

# Create virtual environment directory if it doesn't exist
VENV_DIR="$HOME/.blog_image_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install required packages in the virtual environment
pip install pyyaml requests pillow

# Make the script executable
chmod +x blog_image_setup.py

# Run the script
python blog_image_setup.py "$@"

# Deactivate the virtual environment
deactivate

echo "Script completed. Virtual environment deactivated."
