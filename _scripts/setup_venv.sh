#!/bin/bash

# Create a virtual environment in the _scripts directory if it doesn't exist
if [ ! -d "/Users/saptak/code/saptak.github.io/_scripts/venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv /Users/saptak/code/saptak.github.io/_scripts/venv
fi

# Activate the virtual environment
source /Users/saptak/code/saptak.github.io/_scripts/venv/bin/activate

# Install required packages in the virtual environment
echo "Installing required packages..."
pip install python-frontmatter pyyaml requests

echo "Virtual environment is set up and ready to use."
echo "To activate it manually, run: source /Users/saptak/code/saptak.github.io/_scripts/venv/bin/activate"
