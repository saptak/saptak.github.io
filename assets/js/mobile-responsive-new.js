/**
 * Simple mobile responsiveness enhancements
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
      if (!table.parentElement.classList.contains('table-responsive')) {
        const wrapper = document.createElement('div');
        wrapper.classList.add('table-responsive');
        wrapper.style.overflowX = 'auto';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      }
    });
    
    // Fix for blog post listing page - simple approach
    if (window.location.pathname.includes('/writing/') || 
        document.querySelector('.posts') || 
        document.querySelector('.post-list')) {
      
      // Direct fix for blog post titles in the listing
      const blogTitles = document.querySelectorAll('.post-title, .posts h1, .posts h2, .post-link h1, .post-link h2');
      blogTitles.forEach(title => {
        // Force proper text wrapping
        title.style.display = 'inline-block';
        title.style.width = '100%';
        title.style.maxWidth = '100%';
        title.style.boxSizing = 'border-box';
        title.style.paddingRight = '5px';
        title.style.overflowWrap = 'break-word';
        title.style.wordWrap = 'break-word';
        title.style.wordBreak = 'break-word';
        title.style.hyphens = 'auto';
        title.style.whiteSpace = 'normal';
        
        // Apply styles to parent container if needed
        const parent = title.parentElement;
        if (parent) {
          parent.style.width = '100%';
          parent.style.maxWidth = '100%';
          parent.style.boxSizing = 'border-box';
        }
      });
    }
  }
  
  // Initialize when DOM is loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhanceMobileExperience);
  } else {
    enhanceMobileExperience();
  }
})();
