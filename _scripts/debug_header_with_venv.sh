#!/bin/bash

# Set up and activate the virtual environment
bash /Users/saptak/code/saptak.github.io/_scripts/setup_venv.sh

# Make the Python script executable
chmod +x /Users/saptak/code/saptak.github.io/_scripts/debug_header_images.py

# Run the Python script using the virtual environment Python
echo "Running header diagnostics..."
/Users/saptak/code/saptak.github.io/_scripts/venv/bin/python /Users/saptak/code/saptak.github.io/_scripts/debug_header_images.py

echo "Done! Check the output above for diagnostics."
