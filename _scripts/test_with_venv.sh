#!/bin/bash

# Set up and activate the virtual environment
bash /Users/saptak/code/saptak.github.io/_scripts/setup_venv.sh

# Make the Python scripts executable
chmod +x /Users/saptak/code/saptak.github.io/_scripts/add_unsplash_images.py
chmod +x /Users/saptak/code/saptak.github.io/_scripts/test_one_post.py

# Run the test script using the virtual environment Python
echo "Testing image addition on one post..."
/Users/saptak/code/saptak.github.io/_scripts/venv/bin/python /Users/saptak/code/saptak.github.io/_scripts/test_one_post.py

echo "Done! Check the test post for added images."
