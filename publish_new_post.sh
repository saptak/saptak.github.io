#!/bin/bash

# Make the script executable
chmod +x "$0"

# Check if filename is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <blog-post-filename>"
  echo "Example: $0 2025-03-25-new-blog-post.md"
  exit 1
fi

BLOG_FILE="$1"
BASE_NAME=$(basename "$BLOG_FILE" .md)

# Change to the git repository directory
cd /Users/saptak/code/saptak.github.io

# Ensure the blog post exists
if [ ! -f "_posts/$BLOG_FILE" ]; then
  echo "Error: Blog post file _posts/$BLOG_FILE does not exist."
  exit 1
fi

# Create directories for images if they don't exist
mkdir -p assets/img/blog/headers
mkdir -p assets/img/blog/thumbnails

# Check if images already exist
HEADER_IMAGE="assets/img/blog/headers/$BASE_NAME.jpg"
THUMBNAIL_IMAGE="assets/img/blog/thumbnails/$BASE_NAME.jpg"

if [ ! -f "$HEADER_IMAGE" ] || [ ! -f "$THUMBNAIL_IMAGE" ]; then
  echo "Images for this blog post don't exist yet."
  echo "Creating placeholder images..."
  
  # Create placeholder images
  touch "$HEADER_IMAGE"
  touch "$THUMBNAIL_IMAGE"
  
  echo "Placeholder images created at:"
  echo "- $HEADER_IMAGE"
  echo "- $THUMBNAIL_IMAGE"
  echo ""
  echo "NOTE: You should replace these with proper images later."
fi

# Show current git status
echo "Current git status:"
git status

# Add the blog post and images
echo "Adding blog post and images..."
git add "_posts/$BLOG_FILE"
git add "$HEADER_IMAGE" "$THUMBNAIL_IMAGE"

# Commit the changes
echo "Committing changes..."
POST_TITLE=$(grep -m 1 "title:" "_posts/$BLOG_FILE" | sed 's/title:[[:space:]]*//g' | sed 's/^"\(.*\)"$/\1/g')
git commit -m "Add blog post: $POST_TITLE"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin master

echo "Blog post has been published!"
echo "GitHub Actions will now build and deploy your site."
