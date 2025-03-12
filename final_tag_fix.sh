#!/bin/bash

# Ensure the script is executable
chmod +x "$0"

# Check status before committing
git status

# Add the modified and new files to staging area
git add _includes/post-outline.html
git add _includes/inline-tags.html
git add _includes/inline-tag-script.html
git add _layouts/default.html
git add assets/js/force-inline-tags.js
git add _includes/scripts.html

# Commit changes with a descriptive message
git commit -m "Complete, direct tag fix with inline script and simplified HTML"

# Push changes to the remote repository
git push origin master

echo "Final tag fixes have been committed and pushed to GitHub!"
