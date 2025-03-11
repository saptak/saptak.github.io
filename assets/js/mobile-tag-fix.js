/**
 * Mobile-specific fix for tag display without inline styles
 */
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on mobile
    const isMobile = window.innerWidth <= 768;
    
    if (isMobile) {
      // Add a class to the body for mobile-specific styling
      document.body.classList.add('mobile-view');
      
      // Find all tag containers
      const tagContainers = document.querySelectorAll('.tags-container');
      
      tagContainers.forEach(container => {
        // Add a clearfix class to help with floating elements
        container.classList.add('clearfix-tags');
        
        // Find all tag links in this container
        const tagLinks = container.querySelectorAll('.tag-link');
        tagLinks.forEach((link, index) => {
          // Add mobile-specific class
          link.classList.add('mobile-tag-link');
          
          // Add pill styling for mobile tags
          link.classList.add('tag-pill');
        });
      });
    }
    
    // Handle window resize
    window.addEventListener('resize', function() {
      const isMobileNow = window.innerWidth <= 768;
      if (isMobileNow) {
        document.body.classList.add('mobile-view');
      } else {
        document.body.classList.remove('mobile-view');
      }
    });
  });
})();
