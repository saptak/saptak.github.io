#!/usr/bin/env python3
import os
import re
import yaml
import requests
import random
from urllib.parse import quote_plus
from pathlib import Path
import frontmatter

# Configuration
BLOG_DIR = Path('/Users/saptak/code/saptak.github.io/_posts')
THUMBNAIL_DIR = Path('/Users/saptak/code/saptak.github.io/assets/img/blog/thumbnails')
HEADER_DIR = Path('/Users/saptak/code/saptak.github.io/assets/img/blog/headers')

# Ensure directories exist
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
HEADER_DIR.mkdir(parents=True, exist_ok=True)

# Unsplash API - Using the public URL pattern for free images
UNSPLASH_SEARCH_URL = "https://unsplash.com/s/photos/{}"

def extract_keywords(content):
    """Extract keywords from post title, categories, tags and content."""
    keywords = []
    
    # Add title, category, and tag keywords
    if 'title' in content:
        keywords.extend(content['title'].lower().split())
    
    if 'categories' in content:
        if isinstance(content['categories'], list):
            keywords.extend([cat.lower() for cat in content['categories']])
    
    if 'tags' in content:
        if isinstance(content['tags'], list):
            keywords.extend([tag.lower() for tag in content['tags']])
    
    # Filter out common words and keep meaningful keywords
    stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
                 'as', 'and', 'or', 'of', 'is', 'are', 'was', 'were', 'be', 'been'}
    
    keywords = [k for k in keywords if k.lower() not in stop_words and len(k) > 3]
    
    # Prioritize certain keywords
    priority_keywords = ['technology', 'data', 'machine', 'learning', 'artificial', 'intelligence',
                        'programming', 'software', 'development', 'cloud', 'digital', 'api',
                        'engineering', 'system', 'architecture', 'blockchain', 'algorithm']
    
    # Give priority to technology-related keywords
    keywords = sorted(keywords, key=lambda k: k.lower() in priority_keywords, reverse=True)
    
    # Include a default abstract keyword if nothing else is available
    if not keywords:
        return ['abstract', 'technology']
    
    # Add 'abstract' to make images more visually appealing
    keywords.insert(0, 'abstract')
    
    # Take just the top keywords to make search more specific
    return keywords[:3]

def find_unsplash_image_url(keywords):
    """Find a free image URL from Unsplash based on keywords."""
    search_term = '-'.join(keywords)
    search_url = UNSPLASH_SEARCH_URL.format(quote_plus(search_term))
    
    try:
        # Get the search results page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to retrieve search results for {search_term}. Falling back to abstract.")
            return find_unsplash_image_url(['abstract', 'technology'])
        
        # Extract image URLs from the response
        # This is a very simple pattern that might need to be adjusted based on actual Unsplash HTML structure
        img_pattern = r'https://images\.unsplash\.com/[^\s\'\"]+\?ixlib=rb[^\s\'\"\&]+'
        img_urls = re.findall(img_pattern, response.text)
        
        # Filter for only license=free images
        free_img_urls = [url for url in img_urls if 'license=free' in url or 'license=unsplash' in url]
        
        if not free_img_urls:
            # If no free images, use any image
            if img_urls:
                return random.choice(img_urls)
            else:
                print(f"No images found for {search_term}. Falling back to abstract.")
                return find_unsplash_image_url(['abstract', 'technology'])
        
        return random.choice(free_img_urls)
    
    except Exception as e:
        print(f"Error finding image: {e}")
        # Fallback image if we can't get one from Unsplash
        return "https://images.unsplash.com/photo-1547954575-855750c57bd3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2940&q=80"

def download_image(url, destination):
    """Download an image from URL to destination path."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, stream=True)
        
        if response.status_code == 200:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"Failed to download image: HTTP {response.status_code}")
            return False
    
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def update_post_with_images(post_path):
    """Update a blog post with Unsplash images."""
    try:
        # Read the post with frontmatter
        post = frontmatter.load(post_path)
        
        # Generate base filename from post date and slug
        post_date = post.get('date', '')
        if post_date:
            date_part = post_date.strftime('%Y-%m-%d')
        else:
            date_part = os.path.basename(post_path).split('-', 3)[:3]
            date_part = '-'.join(date_part)
        
        # Create a slug from the post name
        slug = os.path.basename(post_path).split('-', 3)[3].replace('.md', '')
        base_filename = f"{date_part}-{slug}"
        
        # Check if images already exist
        thumbnail_path = THUMBNAIL_DIR / f"{base_filename}.jpg"
        header_path = HEADER_DIR / f"{base_filename}.jpg"
        
        # Skip if already has images defined
        if post.get('thumbnail_path') and post.get('header_image_path'):
            print(f"Post already has images: {post_path}")
            return
        
        # Extract keywords for image search
        keywords = extract_keywords(post)
        print(f"Searching images for {base_filename} with keywords: {keywords}")
        
        # Get image URLs
        image_url = find_unsplash_image_url(keywords)
        
        # Download and save images
        thumbnail_rel_path = f"/assets/img/blog/thumbnails/{base_filename}.jpg"
        header_rel_path = f"/assets/img/blog/headers/{base_filename}.jpg"
        
        if download_image(image_url, thumbnail_path):
            post['thumbnail_path'] = thumbnail_rel_path
        
        if download_image(image_url, header_path):
            post['header_image_path'] = header_rel_path
        
        # Save updated post
        with open(post_path, 'w') as f:
            f.write(frontmatter.dumps(post))
        
        print(f"Updated post with images: {post_path}")
    
    except Exception as e:
        print(f"Error updating post {post_path}: {e}")

def main():
    """Main function to process all blog posts."""
    # Get all markdown files in the blog directory
    post_files = sorted(list(BLOG_DIR.glob('*.md')))
    
    for post_path in post_files:
        print(f"Processing {post_path}...")
        update_post_with_images(post_path)

if __name__ == '__main__':
    main()
