#!/bin/bash

# Make this script executable
chmod +x "$0"

# Change to the git repository directory
cd /Users/saptak/code/saptak.github.io

# Show current git status
echo "Current git status:"
git status

# Pull the latest changes to ensure we're up to date
echo "Pulling latest changes..."
git pull origin master

# Force Jekyll to rebuild by making a small change to the blog post
echo "Making a small change to trigger rebuild..."
current_date=$(date +"%Y-%m-%d %H:%M:%S %z")
sed -i '' "s/date: 2025-03-23/date: $current_date/g" _posts/2025-03-23-mastering-long-term-agentic-memory-with-langgraph.md

# Add the modified file
echo "Adding modified files..."
git add _posts/2025-03-23-mastering-long-term-agentic-memory-with-langgraph.md

# Commit the changes
echo "Committing changes..."
git commit -m "Update blog post timestamp to trigger rebuild"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin master

echo "Blog post has been updated and pushed to GitHub!"
echo "GitHub Pages should rebuild and update the site shortly."