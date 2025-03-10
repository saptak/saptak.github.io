#!/bin/bash

# Remove Gemfile.lock
rm -f Gemfile.lock

# Add and commit changes
git add Gemfile _config.yml
git commit -m "Update Gemfile with compatible dependencies"

# Push changes
git push origin fix-tzinfo-dependency

echo "Changes committed and pushed. Now run: bundle install"
