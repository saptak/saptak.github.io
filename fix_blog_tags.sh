#!/bin/bash
# Fix blog tags using the same Python virtual environment as setup_blog_images_venv.sh

# Create virtual environment directory if it doesn't exist
VENV_DIR="$HOME/.blog_image_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install required packages in the virtual environment if not already installed
pip install pyyaml

# Create temporary Python script
cat > /tmp/fix_blog_tags_temp.py << 'EOF'
#!/usr/bin/env python3
"""
Fix Blog Post Tags Script

This script:
1. Scans all blog posts in the _posts directory
2. Checks and fixes the format of tags and categories in the front matter
3. Ensures that all posts have properly formatted tags

Run this once to fix existing tag issues across your blog.
"""

import os
import re
import yaml
import sys

def extract_front_matter(post_path):
    """Extract front matter from a markdown file."""
    with open(post_path, 'r') as f:
        content = f.read()
    
    # Extract front matter
    front_matter_match = re.match(r'---\n(.*?)\n---', content, re.DOTALL)
    if not front_matter_match:
        print(f"Error: Could not find YAML front matter in {post_path}")
        return None, content
    
    front_matter_yaml = front_matter_match.group(1)
    try:
        front_matter = yaml.safe_load(front_matter_yaml)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in {post_path}: {e}")
        return None, content
    
    return front_matter, content

def update_front_matter(post_path, front_matter, content):
    """Update the front matter in a post."""
    # Fix categories and tags format
    if 'categories' in front_matter and isinstance(front_matter['categories'], str):
        # Convert space-separated string to a list
        front_matter['categories'] = front_matter['categories'].split()
        print(f"  - Fixed categories format in {os.path.basename(post_path)}")
    
    # Ensure tags are present
    if 'categories' in front_matter and 'tags' not in front_matter:
        # Use categories as tags if tags don't exist
        if isinstance(front_matter['categories'], list):
            front_matter['tags'] = front_matter['categories']
        else:
            front_matter['tags'] = [front_matter['categories']]
        print(f"  - Added tags based on categories in {os.path.basename(post_path)}")
    
    # Convert the front matter to YAML
    new_front_matter = yaml.dump(front_matter, default_flow_style=False)
    
    # Replace the original front matter
    new_content = re.sub(r'---\n(.*?)\n---', f'---\n{new_front_matter}---', content, flags=re.DOTALL)
    
    with open(post_path, 'w') as f:
        f.write(new_content)
    
    return True

def main():
    """Main function to run the script."""
    # Get the directory where the script is running
    current_dir = os.getcwd()
    posts_dir = os.path.join(current_dir, "_posts")
    
    if not os.path.exists(posts_dir):
        print(f"Error: Posts directory '{posts_dir}' not found")
        sys.exit(1)
    
    # Get all markdown files in the posts directory
    posts = [os.path.join(posts_dir, f) for f in os.listdir(posts_dir) if f.endswith('.md')]
    
    if not posts:
        print("No blog posts found")
        sys.exit(1)
    
    print(f"Found {len(posts)} blog posts to check")
    
    # Process each post
    fixed_count = 0
    for post_path in posts:
        front_matter, content = extract_front_matter(post_path)
        if front_matter is None:
            continue
        
        # Check if post needs fixing
        needs_fixing = False
        if 'categories' in front_matter and isinstance(front_matter['categories'], str):
            needs_fixing = True
        if 'categories' in front_matter and 'tags' not in front_matter:
            needs_fixing = True
        
        if needs_fixing:
            print(f"Fixing tags in {os.path.basename(post_path)}")
            update_front_matter(post_path, front_matter, content)
            fixed_count += 1
    
    print(f"Fixed tags in {fixed_count} blog posts")

if __name__ == "__main__":
    main()
EOF

# Run the temporary script
cd /Users/saptak/code/saptak.github.io
python /tmp/fix_blog_tags_temp.py

# Clean up
rm /tmp/fix_blog_tags_temp.py

# Deactivate the virtual environment
deactivate

echo "Script completed. Virtual environment deactivated."
