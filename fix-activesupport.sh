#!/bin/bash
cd /Users/saptak/code/saptak.github.io

# Commit the changes to the Gemfile
git add Gemfile
git commit -m "Fix activesupport version to match Gemfile.lock"
git push origin fix-tzinfo-dependency
