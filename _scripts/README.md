# Blog Image Scripts

These scripts help you add free Unsplash images to your blog posts.

## Setup

The Python script requires a few dependencies:

```bash
pip install python-frontmatter pyyaml requests
```

## Usage

1. Make the scripts executable:
   ```bash
   bash _scripts/chmod_scripts.sh
   ```

2. Run the update script:
   ```bash
   bash _scripts/update_blog_images.sh
   ```

This will:
1. Add Unsplash images to all blog posts that don't already have images
2. Commit the changes to Git
3. Push the changes to GitHub

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
