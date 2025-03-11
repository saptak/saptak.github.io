#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Import the functions from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from add_unsplash_images import update_post_with_images

# Test on the AI code generation post
test_post = Path('/Users/saptak/code/saptak.github.io/_posts/2025-03-10-ai-code-generation-tools.md')

if test_post.exists():
    print(f"Testing image addition on: {test_post}")
    update_post_with_images(test_post)
    print("Done!")
else:
    print(f"Test post not found: {test_post}")
