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
  }
  
  // Initialize when DOM is loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhanceMobileExperience);
  } else {
    enhanceMobileExperience();
  }
})();
