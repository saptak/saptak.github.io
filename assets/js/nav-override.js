/**
 * Override the responsive navigation initialization
 * to prevent the hamburger menu from ever being created
 */
(function() {
  // Original responsiveNav function reference
  var originalResponsiveNav = window.responsiveNav;
  
  // Override the function
  window.responsiveNav = function(selector, options) {
    // Add our custom options to disable the hamburger
    var enhancedOptions = options || {};
    
    // Force no custom toggle
    enhancedOptions.customToggle = "";
    
    // Prevent creation of toggle button
    var originalCreateToggle = enhancedOptions.createToggle;
    enhancedOptions._createToggle = function() {
      // Do nothing - don't create the toggle
    };
    
    // Call the original function with our modified options
    var navInstance = originalResponsiveNav(selector, enhancedOptions);
    
    // Force the nav to be always open
    setTimeout(function() {
      // Remove any existing nav toggle
      var existingToggle = document.querySelector('.nav-toggle');
      if (existingToggle && existingToggle.parentNode) {
        existingToggle.parentNode.removeChild(existingToggle);
      }
      
      // Open the navigation and keep it open
      if (navInstance) {
        navInstance.open();
        
        // Override the toggle method to prevent closing
        var originalToggle = navInstance.toggle;
        navInstance.toggle = function() {
          // Always ensure it's open
          navInstance.open();
        };
      }
    }, 100);
    
    return navInstance;
  };
})();
