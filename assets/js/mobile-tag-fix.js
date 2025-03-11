/**
 * Mobile-specific fix for tag display
 */
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on mobile
    const isMobile = window.innerWidth <= 768;
    
    if (isMobile) {
      // Find all tag containers
      const tagContainers = document.querySelectorAll('.tags-container');
      
      tagContainers.forEach(container => {
        // Force inline display
        container.style.display = 'inline';
        container.style.whiteSpace = 'nowrap';
        
        // Find the parent span with the icon
        const parent = container.closest('.mr1.nowrap');
        if (parent) {
          // Make the parent block level for better mobile layout
          parent.style.display = 'block';
          parent.style.width = '100%';
          parent.style.whiteSpace = 'nowrap';
          parent.style.overflowX = 'auto';
          parent.style.marginTop = '0.2rem';
          parent.style.marginBottom = '0.2rem';
        }
      });
      
      // Handle each tag link
      document.querySelectorAll('.tag-link').forEach(link => {
        link.style.display = 'inline';
        link.style.whiteSpace = 'nowrap';
      });
      
      // Handle separators
      document.querySelectorAll('.tag-separator').forEach(sep => {
        sep.style.display = 'inline';
        sep.style.whiteSpace = 'nowrap';
      });
    }
  });
})();
