#!/bin/bash

# First run the image credits script
bash /Users/saptak/code/saptak.github.io/_scripts/add_credits_with_venv.sh

# Then commit all changes
cd /Users/saptak/code/saptak.github.io
git add .
git commit -m "Move header image display to post layout and add Unsplash credits"
git push origin master

echo "All changes have been committed and pushed to GitHub."
