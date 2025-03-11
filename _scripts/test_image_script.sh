#!/bin/bash

# Check for required Python packages and install them if missing
echo "Checking for required Python packages..."
python3 -m pip install python-frontmatter pyyaml requests

# Make the Python scripts executable
chmod +x /Users/saptak/code/saptak.github.io/_scripts/add_unsplash_images.py
chmod +x /Users/saptak/code/saptak.github.io/_scripts/test_one_post.py

# Run the test script
echo "Testing image addition on one post..."
python3 /Users/saptak/code/saptak.github.io/_scripts/test_one_post.py

echo "Done! Check the test post for added images."
