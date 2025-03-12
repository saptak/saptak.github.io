/**
 * Complete removal of hamburger menu from the DOM
 */
(function() {
  function removeHamburger() {
    // Completely remove the hamburger toggle element from the DOM
    const toggles = document.querySelectorAll('.nav-toggle');
    toggles.forEach(function(toggle) {
      if (toggle && toggle.parentNode) {
        toggle.parentNode.removeChild(toggle);
      }
    });
    
    // Force the navigation to stay open
    const navs = document.querySelectorAll('.nav-collapse');
    navs.forEach(function(nav) {
      // Remove closed class
      nav.classList.remove('closed');
      
      // Add opened class
      nav.classList.add('opened');
      
      // Remove aria-hidden attribute
      nav.removeAttribute('aria-hidden');
      
      // Set styles to ensure visibility
      nav.style.position = 'relative';
      nav.style.maxHeight = 'none';
      nav.style.overflow = 'visible';
      nav.style.clip = 'auto';
      nav.style.display = 'block';
    });
    
    // Override the responsive-nav.js behavior
    if (window.responsiveNav) {
      const originalInit = window.responsiveNav;
      window.responsiveNav = function(selector, options) {
        // Modify options to prevent toggle creation
        options = options || {};
        options.customToggle = '';
        options.closeOnNavClick = false;
        
        // Override the create toggle method
        const instance = originalInit(selector, options);
        
        // Keep navigation always open
        setTimeout(function() {
          if (instance && typeof instance.open === 'function') {
            instance.open();
            
            // Override toggle method to always keep it open
            instance.toggle = function() {
              instance.open();
            };
            
            // Override close method
            instance.close = function() {
              // Do nothing - prevent closing
            };
          }
          
          // Remove any remaining toggle buttons
          removeHamburger();
        }, 0);
        
        return instance;
      };
    }
  }
  
  // Run immediately when script loads
  removeHamburger();
  
  // Also run when DOM is fully loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', removeHamburger);
  }
  
  // Run again after a slight delay to catch any dynamically added elements
  setTimeout(removeHamburger, 100);
  
  // Also run after window load
  window.addEventListener('load', removeHamburger);
})();
