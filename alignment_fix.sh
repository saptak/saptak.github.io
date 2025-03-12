#!/bin/bash

# Ensure the script is executable
chmod +x "$0"

# Check status before committing
git status

# Add the modified files to staging area
git add _includes/post-outline.html
git add _includes/inline-tags.html
git add _includes/post-metadata.html

# Commit changes with a descriptive message
git commit -m "Fix alignment issues with date, tags, and comments - ensure icon and text alignment"

# Push changes to the remote repository
git push origin master

echo "Alignment fixes have been committed and pushed to GitHub!"
