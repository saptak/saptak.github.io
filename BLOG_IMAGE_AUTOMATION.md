# Blog Image Automation

This tool automates the process of finding, resizing, and adding images to your blog posts, with a focus on compatibility with macOS environments.

## Quick Start

```bash
# Make the script executable (only needed once)
chmod +x setup_blog_images_venv.sh

# Run with default settings
./setup_blog_images_venv.sh _posts/your-blog-post.md

# Run with custom search terms
./setup_blog_images_venv.sh _posts/your-blog-post.md --search "relevant keywords for your post"
```

This will automatically:
- Find relevant images on Unsplash
- Resize them to the right dimensions for thumbnails and headers
- Add them to your blog repository
- Update your blog post's front matter with the image paths and credits
- Commit and push everything to your GitHub repository

## Prerequisites

- Python 3.6+
- An Unsplash API key (free to obtain at https://unsplash.com/developers)

## Installation

The system has been configured to work with virtual environments on macOS, which is the recommended approach for package management.

1. Ensure the scripts are executable:
   ```bash
   chmod +x setup_blog_images_venv.sh
   chmod +x blog_image_setup.py
   ```

2. (Optional) The Python script has an API key pre-configured, but you can edit it if you want to use your own:
   ```python
   # In blog_image_setup.py
   UNSPLASH_ACCESS_KEY = "YOUR_UNSPLASH_ACCESS_KEY"  # Replace with your key
   ```

## Usage

Use the virtual environment wrapper (recommended for macOS):

```bash
./setup_blog_images_venv.sh PATH_TO_MARKDOWN_FILE [options]
```

This script will:
- Create a Python virtual environment if it doesn't exist
- Install all required packages
- Run the blog image setup script
- Clean up when done

### Options

- `--search "your search terms"`: Specify search terms for Unsplash (default: "code programming ai technology")
- `--no-commit`: Skip the Git commit and push steps
- `--api-key YOUR_KEY`: Provide Unsplash API key via command line instead of using the one in the script

### Example

```bash
./setup_blog_images_venv.sh _posts/2025-03-11-history-of-generative-ai.md --search "generative ai history timeline technology"
```

For specific types of blog posts, try these search term suggestions:

- AI posts: `"artificial intelligence neural network blue technology"`
- Programming: `"code programming developer software computer"`
- Cloud/DevOps: `"cloud computing server network technology"`
- Data Science: `"data visualization analytics charts dashboard"`

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

The settings are located at the top of the Python script (`blog_image_setup.py`).

## Troubleshooting

### Package Installation Issues

If you encounter package installation problems:

1. The virtual environment approach (`setup_blog_images_venv.sh`) should solve most macOS package problems
2. If you still have issues, try installing the required packages manually:
   ```bash
   python3 -m pip install --user pyyaml requests pillow
   ```

### Permission Issues

If you see "Permission denied" errors:

1. Make sure the scripts are executable:
   ```bash
   chmod +x blog_image_setup.py
   chmod +x setup_blog_images_venv.sh
   ```

### Unsplash API Issues

If the script fails to connect to Unsplash:

1. Check that the API key in the script is valid
2. Try using your own API key with the `--api-key` parameter
3. Verify your internet connection

### Git Commit Issues

If the git operations fail:

1. Make sure you're in a git repository
2. Check that the paths to images are correct
3. Use the `--no-commit` flag to skip the git operations and handle them manually
