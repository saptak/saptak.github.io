/**
 * Mobile responsiveness enhancements
 */
(function() {
  // Function to run when the DOM is loaded
  function enhanceMobileExperience() {
    // Add aria attributes to improve accessibility
    const mainContent = document.querySelector('main');
    if (mainContent) {
      mainContent.setAttribute('role', 'main');
      mainContent.setAttribute('id', 'main-content');
    }
    
    // Fix for responsive tables
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
      const wrapper = document.createElement('div');
      wrapper.classList.add('table-responsive');
      wrapper.style.overflowX = 'auto';
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
    
    // Special handling for blog post listing page
    if (window.location.pathname.includes('/writing/') || 
        document.querySelector('.posts') || 
        document.querySelector('.post-list')) {
      
      // Direct fix for text clipping in blog posts
      const fixTextClipping = function() {
        // Target all post elements that might have text clipping
        const postElements = document.querySelectorAll('.post-title, .post-link, .post-meta, .post-summary, .post-date, .post-excerpt');
        
        postElements.forEach(el => {
          // Apply direct anti-clipping fixes
          el.style.overflow = 'visible';
          el.style.paddingRight = '16px';
          el.style.marginRight = '4px';
          el.style.boxSizing = 'content-box';
          el.style.maxWidth = 'calc(100% - 20px)';
          el.style.width = 'auto';
          el.style.wordBreak = 'normal';
          el.style.overflowWrap = 'normal';
          el.style.wordWrap = 'normal';
          
          // Also apply to parent for extra safety
          if (el.parentElement) {
            el.parentElement.style.overflow = 'visible';
            el.parentElement.style.paddingRight = '8px';
          }
        });
        
        // Force proper container sizing
        const containers = document.querySelectorAll('.container, .posts, .post-list, .writing');
        containers.forEach(container => {
          container.style.maxWidth = '100%';
          container.style.paddingRight = '16px';
          container.style.boxSizing = 'border-box';
          container.style.overflow = 'visible';
        });
      };
      
      // Run once on load
      fixTextClipping();
      
      // Run again after a slight delay to catch any dynamic content
      setTimeout(fixTextClipping, 500);
    }
  }
  
  // Initialize when DOM is loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhanceMobileExperience);
  } else {
    enhanceMobileExperience();
  }
})();
