#!/bin/bash

# Check for required Python packages and install them if missing
echo "Checking for required Python packages..."
python3 -m pip install python-frontmatter pyyaml requests

# Make the Python script executable if it's not already
chmod +x /Users/saptak/code/saptak.github.io/_scripts/add_unsplash_images.py

# Run the Python script to add images to blog posts
echo "Adding Unsplash images to blog posts..."
python3 /Users/saptak/code/saptak.github.io/_scripts/add_unsplash_images.py

# Git commands to commit and push changes
echo "Committing changes to Git..."
cd /Users/saptak/code/saptak.github.io
git add .
git commit -m "Add Unsplash images to blog posts"
git push origin master

echo "Done! Changes have been pushed to GitHub."
