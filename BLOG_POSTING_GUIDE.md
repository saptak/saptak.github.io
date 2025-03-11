# Guide to Adding New Blog Posts with Images

This guide explains how to create a new blog post for saptak.github.io with appropriate thumbnail and header images.

## Step 1: Create a New Markdown File

Create a new markdown file in the `_posts` directory with the filename format: `YYYY-MM-DD-post-title.md`

## Step 2: Add Jekyll Front Matter

Each post requires YAML front matter at the beginning of the file. Here's a template:

```yaml
---
layout: post
title: "Your Post Title"
date: YYYY-MM-DD
categories: [primary-category, secondary-category]
tags: [tag1, tag2, tag3]
author: "Saptak"
excerpt: "A brief summary of your post that will appear in preview cards and search results."
thumbnail_path: /assets/img/blog/thumbnails/YYYY-MM-DD-post-title.jpg
header_image_path: /assets/img/blog/headers/YYYY-MM-DD-post-title.jpg
image_credit: "Photo by Photographer Name on Unsplash"
---
```

## Step 3: Find and Add Appropriate Images

### Finding Free Images on Unsplash

1. Visit [Unsplash](https://unsplash.com)
2. Search for relevant, abstract images that relate to your blog topic
3. Look for images that:
   - Have good composition
   - Aren't too busy (especially for thumbnails)
   - Have space for text overlay if needed
   - Match the blog's overall aesthetic

### Recommended Image Topics for Technical Blog Posts:

- Code/Programming: Abstract code, clean workspaces, minimal setups
- AI/Machine Learning: Abstract patterns, neural networks visualizations
- Cloud/Infrastructure: Cloud imagery, server rooms, connected networks
- Data: Data visualizations, clean charts, abstract representations

### Image Sizing Guidelines

- **Thumbnail Images**: 600 x 400px
- **Header Images**: 1200 x 600px

## Step 4: Prepare and Save the Images

1. Download the chosen image from Unsplash
2. Resize the image to the appropriate dimensions
3. Save the images to their respective directories:
   - Thumbnails: `/assets/img/blog/thumbnails/`
   - Headers: `/assets/img/blog/headers/`
4. Use consistent naming: `YYYY-MM-DD-post-title.jpg`

## Step 5: Write Your Blog Content

- Start with a clear introduction
- Use proper Markdown formatting:
  - `#` for main title (already in front matter)
  - `##` for section headings
  - `###` for subsection headings
  - Lists, code blocks, etc. as needed

## Step 6: Properly Credit the Image

Always include proper attribution for Unsplash images:

1. Add the photographer's name in the `image_credit` field in the front matter
2. If possible, include a link to their Unsplash profile somewhere in the post

## Step 7: Commit and Push Your Changes

1. Add the new images and markdown file:
```bash
git add _posts/YYYY-MM-DD-post-title.md
git add assets/img/blog/thumbnails/YYYY-MM-DD-post-title.jpg
git add assets/img/blog/headers/YYYY-MM-DD-post-title.jpg
```

2. Commit your changes:
```bash
git commit -m "Add new blog post: Post Title"
```

3. Push to GitHub:
```bash
git push origin master
```

## Example Unsplash Search Terms for Technical Blogs

- For AI posts: "artificial intelligence", "neural network", "machine learning", "data patterns"
- For code generation: "code", "programming", "developer", "software", "algorithm"
- For cloud services: "cloud computing", "server room", "network", "infrastructure"

## Image Processing Tools

- [Canva](https://www.canva.com/) - Easy online editor
- [GIMP](https://www.gimp.org/) - Free alternative to Photoshop
- [ImageOptim](https://imageoptim.com/) - Helps optimize images for web

---

Remember that high-quality, relevant images can significantly increase the engagement with your blog posts. Choose wisely and ensure they add value to your content!
