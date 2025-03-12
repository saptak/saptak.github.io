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
git commit -m "Fix icon alignment issues using consistent white-space nowrap approach"

# Push changes to the remote repository
git push origin master

echo "Comment icon alignment fixes have been committed and pushed to GitHub!"
