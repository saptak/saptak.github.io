module Jekyll
  class BlogPostGenerator < Generator
    safe true
    
    def generate(site)
      # For each post, create a corresponding blog post
      site.posts.docs.each do |post|
        # Extract the date and slug from the post URL
        date_parts = post.date.strftime('%Y/%m/%d').split('/')
        slug = post.data['slug'] || post.basename_without_ext
        
        # Create a new page for the blog route
        blog_post = PageWithoutAFile.new(site, site.source, File.join('blog', date_parts[0], date_parts[1], date_parts[2]), "#{slug}.html")
        
        # Set the layout and content
        blog_post.data['layout'] = 'blog_post'
        blog_post.data['title'] = post.data['title']
        blog_post.data['date'] = post.date
        blog_post.data['original_post'] = post
        blog_post.data['permalink'] = "/blog/#{date_parts[0]}/#{date_parts[1]}/#{date_parts[2]}/#{slug}"
        
        # Copy other front matter
        %w(tags categories description header_image_path image_credit).each do |key|
          blog_post.data[key] = post.data[key] if post.data.key?(key)
        end
        
        # Set the content to be the same as the original post
        blog_post.content = post.content
        
        # Add the page to the site
        site.pages << blog_post
      end
    end
  end
end
