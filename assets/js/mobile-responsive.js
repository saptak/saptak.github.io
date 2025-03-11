/**
 * Mobile responsiveness enhancements
 */
(function() {
  // Function to run when the DOM is loaded
  function enhanceMobileExperience() {
    // Add touch-friendly tap targets
    const navLinks = document.querySelectorAll('nav a, .button');
    navLinks.forEach(link => {
      // Add a small delay to avoid accidental double taps
      link.addEventListener('touchend', function(e) {
        if (!link.classList.contains('no-delay') && !link.hasAttribute('data-no-delay')) {
          e.preventDefault();
          setTimeout(function() {
            window.location = link.href;
          }, 100);
        }
      });
    });
    
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
      
      // Ensure no horizontal scrolling is possible
      document.documentElement.style.overflowX = 'hidden';
      document.body.style.overflowX = 'hidden';
      
      // Find all blog post titles and ensure they can wrap
      const postTitles = document.querySelectorAll('.post-title, .post-link');
      postTitles.forEach(title => {
        title.style.wordBreak = 'break-word';
        title.style.overflowWrap = 'break-word';
        title.style.width = '100%';
        title.style.maxWidth = '100%';
        title.style.display = 'block';
      });
      
      // Process all links in posts to ensure they wrap
      const postLinks = document.querySelectorAll('.post a, .posts a');
      postLinks.forEach(link => {
        link.style.wordBreak = 'break-word';
        link.style.overflowWrap = 'break-word';
        link.style.maxWidth = '100%';
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
