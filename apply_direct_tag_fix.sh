#!/bin/bash

# Ensure the script is executable
chmod +x "$0"

# Check status before committing
git status

# Add the modified and new files to staging area
git add _includes/post-metadata.html
git add _includes/tag-fix-styles.html
git add _layouts/default.html

# Commit changes with a descriptive message
git commit -m "Apply direct tag fix with inline styles for consistent display"

# Push changes to the remote repository
git push origin master

echo "Direct tag fixes have been committed and pushed to GitHub!"
