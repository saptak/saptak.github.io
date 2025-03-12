#!/bin/bash

# Make the script executable
chmod +x "$0"

# Change to the git repository directory
cd /Users/saptak/code/saptak.github.io

# Show current git status
echo "Current git status:"
git status

# Add all modified files
echo "Adding modified files..."
git add .

# Commit the changes
echo "Committing changes..."
git commit -m "Remove hamburger menu and display dark mode button and blog label directly"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin master

echo "Changes published to GitHub!"
