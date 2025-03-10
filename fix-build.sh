#!/bin/bash
cd /Users/saptak/code/saptak.github.io

# Build locally to test
PAGES_REPO_NWO=saptak/saptak.github.io bundle exec jekyll build --trace

# Add all the fixed files
git add Gemfile _sass/* atom.xml _config.yml .github/workflows/jekyll.yml

# Commit and push
git commit -m "Fix Jekyll build issues with complete SCSS placeholders"
git push origin fix-tzinfo-dependency
