#!/bin/bash
# Simple wrapper script for blog_image_setup.py

# Make the Python script executable
chmod +x blog_image_setup.py

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

# Check for required Python packages
REQUIRED_PACKAGES="pyyaml requests pillow"
for package in $REQUIRED_PACKAGES; do
    if ! python3 -c "import $package" &> /dev/null; then
        echo "Installing required Python package: $package"
        pip3 install $package
    fi
done

# Run the Python script with all arguments passed to this script
python3 blog_image_setup.py "$@"
