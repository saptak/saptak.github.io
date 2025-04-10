#!/usr/bin/env python3
"""
Blog Post Image Update Script

This script:
1. Takes a blog post and updates its images with new ones from Unsplash
2. Ensures the new image is different from the current one
3. Resizes them to recommended dimensions
4. Updates the front matter with new image credits
5. Optionally commits and pushes the changes to the repository
"""

import os
import sys
import re
import random
import subprocess
import argparse
import yaml
import requests
from PIL import Image
from io import BytesIO
import datetime

# Configuration
THUMBNAIL_SIZE = (600, 400)
HEADER_SIZE = (1200, 600)
UNSPLASH_ACCESS_KEY = "uX8MnVBy2-XxgN7i1Qclr9ysQ3s5p8N2zWtv5iIEw0E"  # Replace with your Unsplash API key
GIT_COMMIT_MESSAGE_TEMPLATE = "Update images for blog post: {}"

def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Update blog post images')
    parser.add_argument('post_path', help='Path to the blog post markdown file')
    parser.add_argument('--search', help='Search terms for Unsplash (defaults to keywords from post)')
    parser.add_argument('--no-commit', action='store_true', 
                        help='Skip Git commit and push steps')
    parser.add_argument('--api-key', help='Unsplash API key (overrides hardcoded key)')
    parser.add_argument('--force', action='store_true',
                        help='Force update even if no current images exist')
    return parser.parse_args()

def extract_post_info(post_path):
    """Extract post information from the markdown file."""
    if not os.path.exists(post_path):
        print(f"Error: Post file {post_path} not found")
        sys.exit(1)
    
    with open(post_path, 'r') as f:
        content = f.read()
    
    # Extract front matter
    front_matter_match = re.match(r'---\n(.*?)\n---', content, re.DOTALL)
    if not front_matter_match:
        print("Error: Could not find YAML front matter in the post")
        sys.exit(1)
    
    front_matter = yaml.safe_load(front_matter_match.group(1))
    
    # Extract post filename without extension
    post_filename = os.path.basename(post_path)
    post_name = os.path.splitext(post_filename)[0]
    
    # Extract current image info if available
    current_image_id = None
    if 'image_credit' in front_matter:
        # Try to extract Unsplash image ID from credit URL if it exists
        credit_url = front_matter.get('image_credit_url', '')
        if 'unsplash.com' in credit_url:
            match = re.search(r'unsplash.com/photos/([a-zA-Z0-9_-]+)', credit_url)
            if match:
                current_image_id = match.group(1)
    
    # Extract keywords from post
    keywords = []
    if 'categories' in front_matter:
        if isinstance(front_matter['categories'], list):
            keywords.extend(front_matter['categories'])
        else:
            keywords.append(front_matter['categories'])
    
    if 'tags' in front_matter:
        if isinstance(front_matter['tags'], list):
            keywords.extend(front_matter['tags'])
        else:
            keywords.append(front_matter['tags'])
    
    # Extract title words
    if 'title' in front_matter:
        title_words = re.findall(r'\w+', front_matter['title'])
        keywords.extend([word for word in title_words if len(word) > 3])
    
    return {
        'post_path': post_path,
        'front_matter': front_matter,
        'post_name': post_name,
        'title': front_matter.get('title', 'Blog Post'),
        'current_image_id': current_image_id,
        'keywords': list(set(keywords))  # Remove duplicates
    }

def search_unsplash(search_terms, api_key, exclude_id=None):
    """Search Unsplash for images matching the search terms, excluding current image."""
    if not api_key:
        print("Error: Unsplash API key not provided")
        sys.exit(1)
    
    url = "https://api.unsplash.com/search/photos"
    headers = {
        "Authorization": f"Client-ID {api_key}"
    }
    params = {
        "query": search_terms,
        "per_page": 30,
        "orientation": "landscape"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'results' not in data or not data['results']:
            print(f"No images found for search terms: {search_terms}")
            sys.exit(1)
        
        # Filter out current image if needed
        results = data['results']
        if exclude_id:
            results = [img for img in results if img['id'] != exclude_id]
            
        if not results:
            print(f"No new images found that differ from current image.")
            print(f"Trying another search with modified terms...")
            
            # Try a slightly modified search
            modified_terms = f"{search_terms} photography"
            return search_unsplash(modified_terms, api_key, exclude_id)
        
        # Pick a random image from results
        image = random.choice(results)
        return {
            'thumbnail_url': image['urls']['small'],
            'header_url': image['urls']['regular'],
            'credit': f"Photo by {image['user']['name']} on Unsplash",
            'credit_url': image['links']['html'],
            'id': image['id']
        }
    except requests.exceptions.RequestException as e:
        print(f"Error searching Unsplash: {e}")
        sys.exit(1)

def download_and_resize_image(url, size, save_path):
    """Download an image and resize it to the specified dimensions."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB (in case of PNG with transparency)
        img = img.resize(size, Image.LANCZOS)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the image
        img.save(save_path, 'JPEG', quality=85)
        print(f"Saved image to {save_path}")
        return True
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

def update_front_matter(post_info, image_paths, image_info):
    """Update the post's front matter with image paths and credit."""
    with open(post_info['post_path'], 'r') as f:
        content = f.read()
    
    front_matter = post_info['front_matter']
    front_matter['thumbnail_path'] = image_paths['thumbnail']
    front_matter['header_image_path'] = image_paths['header']
    front_matter['image_credit'] = image_info['credit']
    front_matter['image_credit_url'] = image_info['credit_url']
    
    # Convert the front matter to YAML
    new_front_matter = yaml.dump(front_matter, default_flow_style=False)
    
    # Replace the original front matter
    new_content = re.sub(r'---\n(.*?)\n---', f'---\n{new_front_matter}---', content, flags=re.DOTALL)
    
    with open(post_info['post_path'], 'w') as f:
        f.write(new_content)
    
    print(f"Updated front matter in {post_info['post_path']}")
    return True

def git_operations(post_info, image_paths):
    """Commit and push changes to Git repository."""
    try:
        # Add changes to Git
        subprocess.run(['git', 'add', post_info['post_path']], check=True)
        subprocess.run(['git', 'add', image_paths['thumbnail']], check=True)
        subprocess.run(['git', 'add', image_paths['header']], check=True)
        
        # Commit changes
        commit_message = GIT_COMMIT_MESSAGE_TEMPLATE.format(post_info['title'])
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        
        # Push changes
        subprocess.run(['git', 'push', 'origin', 'master'], check=True)
        
        print("Successfully committed and pushed changes to the repository")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in Git operations: {e}")
        return False

def main():
    """Main function to run the script."""
    args = setup_argparse()
    
    # Extract post information
    post_info = extract_post_info(args.post_path)
    
    # Define image paths
    blog_dir = os.path.dirname(os.path.dirname(args.post_path))
    image_paths = {
        'thumbnail': f"/assets/img/blog/thumbnails/{post_info['post_name']}.jpg",
        'header': f"/assets/img/blog/headers/{post_info['post_name']}.jpg"
    }
    
    # Get full paths for saving
    full_image_paths = {
        'thumbnail': os.path.join(blog_dir, image_paths['thumbnail'].lstrip('/')),
        'header': os.path.join(blog_dir, image_paths['header'].lstrip('/'))
    }
    
    # Check if images already exist
    if not args.force:
        thumbnail_exists = os.path.exists(full_image_paths['thumbnail'])
        header_exists = os.path.exists(full_image_paths['header'])
        
        if not (thumbnail_exists and header_exists):
            print("Warning: One or more images don't exist. Consider running setup_blog_images_venv.sh first.")
            user_input = input("Continue anyway? (y/n): ")
            if user_input.lower() != 'y':
                print("Operation cancelled.")
                sys.exit(0)
    
    # Determine search terms
    search_terms = args.search
    if not search_terms:
        # Use extracted keywords if no search terms provided
        if post_info['keywords']:
            # Limit to 5 keywords to avoid too narrow searches
            keywords = post_info['keywords'][:5]
            search_terms = ' '.join(keywords)
        else:
            search_terms = "code programming technology"
    
    print(f"Using search terms: {search_terms}")
    
    # Search Unsplash for images, excluding current image
    unsplash_api_key = args.api_key or UNSPLASH_ACCESS_KEY
    image_info = search_unsplash(search_terms, unsplash_api_key, post_info['current_image_id'])
    
    # Download and resize images
    print("Downloading and resizing images...")
    download_and_resize_image(image_info['thumbnail_url'], THUMBNAIL_SIZE, full_image_paths['thumbnail'])
    download_and_resize_image(image_info['header_url'], HEADER_SIZE, full_image_paths['header'])
    
    # Update front matter
    print("Updating post front matter...")
    update_front_matter(post_info, image_paths, image_info)
    
    # Git operations
    if not args.no_commit:
        print("Performing Git operations...")
        git_operations(post_info, image_paths)
    
    print(f"\nImage update complete for post: {post_info['title']}")
    print(f"Image credit: {image_info['credit']}")
    print(f"Photo URL: {image_info['credit_url']}")
    
    if post_info['current_image_id']:
        print(f"Replaced previous image ID: {post_info['current_image_id']}")
    print(f"New image ID: {image_info['id']}")

if __name__ == "__main__":
    main()
