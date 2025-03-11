#!/bin/bash

# Change to the project root directory
cd /Users/saptak/code/saptak.github.io

# Make all shell scripts executable
chmod +x _scripts/*.sh

# Commit the changes
git add .
git commit -m "Add virtual environment support for blog image scripts"
git push origin master

echo "Virtual environment support added and changes pushed to GitHub."
