#!/bin/bash

# Change to the project root directory
cd /Users/saptak/code/saptak.github.io

# Commit the changes
git add .
git commit -m "Fix image paths in blog thumbnails"
git push origin master

echo "Image path fixes committed and pushed to GitHub."
