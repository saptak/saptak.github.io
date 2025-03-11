#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Try to import required modules with helpful error messages
try:
    import frontmatter
except ImportError:
    print("Error: 'python-frontmatter' module is not installed.")
    print("Please install it with: pip install python-frontmatter")
    sys.exit(1)

# Configuration
BLOG_DIR = Path('/Users/saptak/code/saptak.github.io/_posts')

def add_image_credits_to_post(post_path):
    """Add image credit to a blog post's frontmatter."""
    try:
        # Read the post with frontmatter
        post = frontmatter.load(post_path)
        
        # Skip if already has image_credit
        if 'image_credit' in post and post['image_credit']:
            print(f"Skipping: Post already has image credit: {post_path.name}")
            return False
        
        # Add credit if post has header image
        if 'header_image_path' in post and post['header_image_path']:
            post['image_credit'] = "Photo by Unsplash"
            
            # Save updated post
            with open(post_path, 'w') as f:
                f.write(frontmatter.dumps(post))
            
            print(f"✅ Added image credit to: {post_path.name}")
            return True
        else:
            print(f"Skipping: Post has no header image: {post_path.name}")
            return False
    
    except Exception as e:
        print(f"❌ Error updating post {post_path.name}: {e}")
        return False

def main():
    """Main function to process all blog posts."""
    # Get all markdown files in the blog directory
    post_files = sorted(list(BLOG_DIR.glob('*.md')))
    
    if not post_files:
        print(f"No blog posts found in {BLOG_DIR}. Please check the directory.")
        return
    
    print(f"Found {len(post_files)} blog posts to process.")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for post_path in post_files:
        try:
            if add_image_credits_to_post(post_path):
                success_count += 1
            else:
                skip_count += 1
        except Exception as e:
            print(f"Error processing post {post_path.name}: {e}")
            error_count += 1
    
    print(f"\nCompleted processing {len(post_files)} blog posts:")
    print(f"- {success_count} posts updated with image credits")
    print(f"- {skip_count} posts skipped")
    print(f"- {error_count} posts encountered errors")

if __name__ == '__main__':
    main()
