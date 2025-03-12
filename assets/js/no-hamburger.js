/**
 * Override the responsive navigation behavior to always show the navigation items
 * and remove the hamburger menu
 */
(function() {
  // Run after the DOM is fully loaded
  function init() {
    // Force the nav to stay open
    var navCollapse = document.querySelector('.nav-collapse');
    if (navCollapse) {
      // Remove closed class and add opened class
      navCollapse.classList.remove('closed');
      navCollapse.classList.add('opened');
      
      // Remove aria-hidden attribute to ensure accessibility
      navCollapse.removeAttribute('aria-hidden');
      
      // Make sure it has proper positioning
      navCollapse.style.position = 'relative';
      navCollapse.style.maxHeight = 'none';
      navCollapse.style.overflow = 'visible';
      
      // Hide the nav toggle button if it exists
      var navToggle = document.querySelector('.nav-toggle');
      if (navToggle) {
        navToggle.style.display = 'none';
      }
    }
  }
  
  // Run immediately and also after the DOM is loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  // Also run after a slight delay to ensure everything is properly initialized
  setTimeout(init, 500);
})();
