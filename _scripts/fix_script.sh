#!/bin/bash

# Change to the project root directory
cd /Users/saptak/code/saptak.github.io

# Make all scripts executable
chmod +x _scripts/*.py
chmod +x _scripts/*.sh

# Commit the changes
git add .
git commit -m "Fix blog image script with better error handling and improved image selection"

# Push changes
git push origin master

echo "Fixes committed and pushed to GitHub."
