#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Try to import required modules with helpful error messages
try:
    import frontmatter
    import yaml
except ImportError:
    print("Error: Required modules are not installed.")
    print("Please install them with: pip install python-frontmatter pyyaml")
    sys.exit(1)

# Configuration
BLOG_DIR = Path('/Users/saptak/code/saptak.github.io/_posts')
LAYOUTS_DIR = Path('/Users/saptak/code/saptak.github.io/_layouts')
INCLUDES_DIR = Path('/Users/saptak/code/saptak.github.io/_includes')

def check_header_setup():
    """Check the setup of headers in the Jekyll site."""
    
    try:
        # Read the layout files
        post_layout = None
        default_layout = None
        
        if (LAYOUTS_DIR / 'post.html').exists():
            with open(LAYOUTS_DIR / 'post.html', 'r') as f:
                post_layout = f.read()
                
        if (LAYOUTS_DIR / 'default.html').exists():
            with open(LAYOUTS_DIR / 'default.html', 'r') as f:
                default_layout = f.read()
        
        # Read the include files
        header_include = None
        post_header_include = None
        
        if (INCLUDES_DIR / 'header.html').exists():
            with open(INCLUDES_DIR / 'header.html', 'r') as f:
                header_include = f.read()
                
        if (INCLUDES_DIR / 'post_header.html').exists():
            with open(INCLUDES_DIR / 'post_header.html', 'r') as f:
                post_header_include = f.read()
        
        # Print the diagnostics
        print("=== HEADER SETUP DIAGNOSTICS ===")
        
        print("\nPost Layout:")
        if post_layout:
            frontmatter_data = yaml.safe_load(post_layout.split('---')[1]) if '---' in post_layout else {}
            print(f"  - include_header: {frontmatter_data.get('include_header', 'Not set')}")
        else:
            print("  - Not found")
            
        print("\nDefault Layout:")
        if default_layout:
            if "{% if page.include_header %}" in default_layout:
                print("  - Checks for page.include_header")
            else:
                print("  - Does NOT check for page.include_header")
                
            if "{% include header.html %}" in default_layout:
                print("  - Includes header.html directly")
            elif "{% include {{ page.include_header }} %}" in default_layout:
                print("  - Includes dynamic header via page.include_header")
        else:
            print("  - Not found")
            
        print("\nHeader Include:")
        if header_include:
            if "{% include {{ page.include_header }} %}" in header_include:
                print("  - Dynamically includes page.include_header")
            else:
                print("  - Does NOT dynamically include page.include_header")
        else:
            print("  - Not found")
            
        print("\nPost Header Include:")
        if post_header_include:
            if "{% if page.header_image_path %}" in post_header_include:
                print("  - Checks for page.header_image_path")
            else:
                print("  - Does NOT check for page.header_image_path")
        else:
            print("  - Not found")
        
        # Check a few blog posts
        print("\n=== BLOG POST DIAGNOSTICS ===")
        
        post_files = sorted(list(BLOG_DIR.glob('*.md')))[:3]  # Just check first 3 posts
        
        for post_path in post_files:
            post = frontmatter.load(post_path)
            print(f"\nPost: {post_path.name}")
            print(f"  - layout: {post.get('layout', 'Not set')}")
            print(f"  - include_header: {post.get('include_header', 'Not set')}")
            print(f"  - header_image_path: {post.get('header_image_path', 'Not set')}")
        
    except Exception as e:
        print(f"Error during diagnostics: {str(e)}")

if __name__ == '__main__':
    check_header_setup()
