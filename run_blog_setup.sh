#!/bin/bash

# Make the script executable
chmod +x "$0"

# Change to the git repository directory
cd /Users/saptak/code/saptak.github.io

# Make the setup script executable
chmod +x setup_blog_images_venv.sh
chmod +x blog_image_setup.py

# Run the setup script with appropriate search terms
echo "Running setup_blog_images_venv.sh to download images and publish blog..."
python blog_image_setup.py _posts/2025-03-13-integrating-openai-responses-api-with-google-gemini.md --search "artificial intelligence api integration technology bridge connection"

echo "Blog post has been published with images!"
