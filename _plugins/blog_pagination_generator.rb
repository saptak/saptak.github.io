module Jekyll
  class BlogPaginationGenerator < Generator
    safe true

    def generate(site)
      # Get the total number of posts
      total_posts = site.posts.docs.size
      posts_per_page = site.config['paginate'] || 10

      # Calculate the number of pages needed
      total_pages = (total_posts.to_f / posts_per_page).ceil

      # Generate pagination pages (starting from page 2, since page 1 is index.html)
      (2..total_pages).each do |page_num|
        # Create a new page for each pagination page
        pagination_page = PageWithoutAFile.new(site, site.source, File.join('blog', "page#{page_num}"), "index.html")

        # Set the layout and content
        pagination_page.data['layout'] = 'default'
        pagination_page.data['title'] = "Saptak Sen: Blog - Page #{page_num}"
        pagination_page.data['nav_item'] = 'blog'
        pagination_page.data['nav_item_writing'] = 'all'
        pagination_page.data['header_title'] = 'Blog'
        pagination_page.data['include_header'] = 'blog_header.html'
        pagination_page.data['page_num'] = page_num
        pagination_page.data['include_comments'] = { 'count' => true }

        # Use the same content as blog/index.html but don't try to parse front matter
        content = File.read(File.join(site.source, 'blog', 'index.html'))
        if content =~ /\A(---\s*\n.*?\n?)^((---|\.\.\.)\s*$\n?)/m
          content = $'
        end
        pagination_page.content = content

        # Add the page to the site
        site.pages << pagination_page
      end
    end
  end
end
