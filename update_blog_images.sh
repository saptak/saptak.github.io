#!/bin/bash
# Update blog images with new Unsplash images
# Usage: ./update_blog_images.sh path/to/blogpost.md [search_terms]

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
chmod +x update_blog_images.py

# Run the script
python update_blog_images.py "$@"

# Deactivate the virtual environment
deactivate

echo "Script completed. Virtual environment deactivated."
