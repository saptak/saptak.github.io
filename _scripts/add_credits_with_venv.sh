#!/bin/bash

# Set up and activate the virtual environment
bash /Users/saptak/code/saptak.github.io/_scripts/setup_venv.sh

# Make the Python script executable
chmod +x /Users/saptak/code/saptak.github.io/_scripts/add_image_credits.py

# Run the Python script using the virtual environment Python
echo "Adding image credits to blog posts..."
/Users/saptak/code/saptak.github.io/_scripts/venv/bin/python /Users/saptak/code/saptak.github.io/_scripts/add_image_credits.py

# Commit the changes
echo "Committing changes to Git..."
cd /Users/saptak/code/saptak.github.io
git add .
git commit -m "Add image credits to blog posts and fix header image display"
git push origin master

echo "Done! Changes have been pushed to GitHub."
