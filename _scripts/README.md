# Blog Image Scripts

These scripts help you add free Unsplash images to your blog posts.

## Setup

The Python script requires a few dependencies that will be installed in a virtual environment:

```bash
bash setup_venv.sh
```

This will create a virtual environment in the `_scripts/venv` directory and install all required packages.

## Usage

1. Test the script on a single post:
   ```bash
   bash test_with_venv.sh
   ```

2. Run the script on all blog posts:
   ```bash
   bash update_with_venv.sh
   ```

This will:
1. Set up a Python virtual environment with all dependencies
2. Add Unsplash images to all blog posts that don't already have images
3. Commit the changes to Git
4. Push the changes to GitHub

## How It Works

The script:
- Extracts keywords from your blog post (title, tags, categories)
- Searches Unsplash for free images matching those keywords
- Downloads the images to your project's image directories
- Updates the blog post frontmatter to include the image paths
- Commits and pushes the changes to GitHub

## Image Placement

Images will appear in two places:
1. As thumbnails in the blog list view
2. As header images at the top of each blog post
