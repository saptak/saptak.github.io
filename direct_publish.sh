#!/bin/bash

# Make this script executable
chmod +x "$0"

# Change to the git repository directory
cd /Users/saptak/code/saptak.github.io

# Ensure directories exist
mkdir -p assets/img/blog/headers
mkdir -p assets/img/blog/thumbnails

# Create placeholder image files
echo "Creating placeholder images..."
echo "placeholder" > assets/img/blog/headers/2025-03-13-integrating-openai-responses-api-with-google-gemini.jpg
echo "placeholder" > assets/img/blog/thumbnails/2025-03-13-integrating-openai-responses-api-with-google-gemini.jpg

# Show current git status
echo "Current git status:"
git status

# Add all modified files
echo "Adding modified files..."
git add _posts/2025-03-13-integrating-openai-responses-api-with-google-gemini.md
git add assets/img/blog/headers/2025-03-13-integrating-openai-responses-api-with-google-gemini.jpg
git add assets/img/blog/thumbnails/2025-03-13-integrating-openai-responses-api-with-google-gemini.jpg

# Commit the changes
echo "Committing changes..."
git commit -m "Add blog post: Integrating OpenAI Responses API with Google Gemini"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin master

echo "Blog post has been published with placeholder images!"
echo "NOTE: You may want to replace the placeholder images with actual images later."
