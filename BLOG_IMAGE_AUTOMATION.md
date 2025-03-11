# Blog Image Automation

This tool automates the process of finding, resizing, and adding images to your blog posts.

## Prerequisites

- Python 3.6+
- An Unsplash API key (free to obtain at https://unsplash.com/developers)
- Required Python packages: `pyyaml`, `requests`, `pillow` (the script will attempt to install these)

## Setup

1. Edit the Python script `blog_image_setup.py` and add your Unsplash API key:
   ```python
   UNSPLASH_ACCESS_KEY = "YOUR_UNSPLASH_ACCESS_KEY"  # Replace with your key
   ```

2. Make the shell script executable:
   ```bash
   chmod +x setup_blog_images.sh
   ```

## Usage

```bash
./setup_blog_images.sh PATH_TO_MARKDOWN_FILE [options]
```

### Options

- `--search "your search terms"`: Specify search terms for Unsplash (default: "code programming ai technology")
- `--no-commit`: Skip the Git commit and push steps
- `--api-key YOUR_KEY`: Provide Unsplash API key via command line instead of hardcoding

### Example

```bash
./setup_blog_images.sh _posts/2025-03-10-ai-code-generation-tools.md --search "code artificial intelligence programming"
```

## What the Script Does

1. Extracts information from your blog post's front matter
2. Searches Unsplash for images matching your search terms
3. Downloads and resizes the images:
   - Thumbnail: 600 x 400 pixels
   - Header: 1200 x 600 pixels
4. Saves the images to the appropriate directories:
   - `/assets/img/blog/thumbnails/`
   - `/assets/img/blog/headers/`
5. Updates your post's front matter with the image paths and proper attribution
6. Commits and pushes the changes to your repository

## Customization

You can modify the script to change:
- Image dimensions
- Default search terms
- Commit message format
- Image quality settings

The settings are located at the top of the Python script.
