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

# Import the functions from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from add_unsplash_images import update_post_with_images
except ImportError as e:
    print(f"Error importing from add_unsplash_images.py: {e}")
    print("Please make sure the script exists and all its dependencies are installed.")
    sys.exit(1)

# Test on the AI code generation post
test_post = Path('/Users/saptak/code/saptak.github.io/_posts/2025-03-10-ai-code-generation-tools.md')

if test_post.exists():
    print(f"Testing image addition on: {test_post}")
    try:
        # Check if the post already has images
        post = frontmatter.load(test_post)
        if post.get('thumbnail_path') and post.get('header_image_path'):
            print("This post already has images. Removing them to test again...")
            post.pop('thumbnail_path', None)
            post.pop('header_image_path', None)
            with open(test_post, 'w') as f:
                f.write(frontmatter.dumps(post))
        
        update_post_with_images(test_post)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
else:
    print(f"Test post not found: {test_post}")
    print("Available posts:")
    blog_dir = Path('/Users/saptak/code/saptak.github.io/_posts')
    for post in blog_dir.glob('*.md'):
        print(f"  - {post.name}")
    sys.exit(1)
