#!/bin/bash

# Make the script executable
chmod +x "$0"

# Get commit message from arguments or prompt for one
if [ $# -eq 0 ]; then
  read -p "Enter commit message: " COMMIT_MESSAGE
else
  COMMIT_MESSAGE="$*"
fi

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
git commit -m "$COMMIT_MESSAGE"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin master

echo "Changes published to GitHub!"
echo "GitHub Actions will now build and deploy your site."
