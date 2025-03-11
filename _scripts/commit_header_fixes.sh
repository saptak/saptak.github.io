#!/bin/bash

# Change to the project root directory
cd /Users/saptak/code/saptak.github.io

# Make all scripts executable
chmod +x _scripts/*.py
chmod +x _scripts/*.sh

# Commit the changes
git add .
git commit -m "Fix header image display and add image credits"
git push origin master

echo "Header image fixes committed and pushed to GitHub."
