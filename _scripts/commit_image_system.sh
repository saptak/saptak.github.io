#!/bin/bash

# Change to the project root directory
cd /Users/saptak/code/saptak.github.io

# Make all scripts executable
chmod +x _scripts/*.py
chmod +x _scripts/*.sh

# Commit the changes
git add .
git commit -m "Add blog image system with Unsplash integration"

echo "Changes committed. You can now push with: git push origin master"
