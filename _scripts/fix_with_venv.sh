#!/bin/bash

# Make all scripts executable
chmod +x /Users/saptak/code/saptak.github.io/_scripts/*.sh

# Commit the changes
cd /Users/saptak/code/saptak.github.io
git add .
git commit -m "Add virtual environment support for blog image scripts"
git push origin master

echo "Virtual environment support added. Changes committed and pushed to GitHub."
