/**
 * Ensures proper alignment of header elements including dark mode button and navigation items
 */
(function() {
  function fixHeaderAlignment() {
    // Get the darkMode toggle button and the blog button
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const blogButton = document.querySelector('.nav-button.caps');
    const siteNameLink = document.querySelector('.left .nav-button');
    
    if (!darkModeToggle) return;
    
    // Fix site name alignment
    if (siteNameLink) {
      siteNameLink.style.display = 'flex';
      siteNameLink.style.alignItems = 'center';
      siteNameLink.style.justifyContent = 'center';
      siteNameLink.style.height = '100%';
      siteNameLink.style.paddingTop = '0';
      siteNameLink.style.paddingBottom = '0';
      siteNameLink.style.lineHeight = 'normal';
      
      // Also fix the span inside
      const siteNameSpan = siteNameLink.querySelector('span');
      if (siteNameSpan) {
        siteNameSpan.style.display = 'inline-flex';
        siteNameSpan.style.alignItems = 'center';
        siteNameSpan.style.verticalAlign = 'middle';
        siteNameSpan.style.lineHeight = 'normal';
      }
      
      // Fix the parent div
      const leftDiv = document.querySelector('nav[role="navigation"] .left');
      if (leftDiv) {
        leftDiv.style.display = 'flex';
        leftDiv.style.alignItems = 'center';
        leftDiv.style.height = '100%';
      }
    }
    
    // Fix overall navigation
    const mainNav = document.querySelector('nav[role="navigation"]');
    if (mainNav) {
      mainNav.style.display = 'flex';
      mainNav.style.alignItems = 'center';
      mainNav.style.minHeight = '56px';
    }
    
    // Check for any flex layout issues
    const navCollapse = document.querySelector('.nav-collapse');
    if (navCollapse) {
      const navCollapseUl = navCollapse.querySelector('ul');
      if (navCollapseUl) {
        // Ensure flex display is set
        navCollapseUl.style.display = 'flex';
        navCollapseUl.style.alignItems = 'center';
        
        // Get all list items and set them to flex as well
        const listItems = navCollapseUl.querySelectorAll('li');
        listItems.forEach(li => {
          li.style.display = 'flex';
          li.style.alignItems = 'center';
        });
      }
    }
    
    // Ensure the dark mode toggle button is aligned
    if (darkModeToggle) {
      // Set button styles
      darkModeToggle.style.display = 'flex';
      darkModeToggle.style.alignItems = 'center';
      darkModeToggle.style.justifyContent = 'center';
      darkModeToggle.style.verticalAlign = 'middle';
      darkModeToggle.style.margin = '0';
      darkModeToggle.style.padding = '8px';
      
      // Get the parent list item and ensure it's styled consistently
      const toggleLi = darkModeToggle.closest('li');
      if (toggleLi) {
        toggleLi.style.display = 'flex';
        toggleLi.style.alignItems = 'center';
        toggleLi.style.height = '100%';
      }
      
      // Fix the icon inside the button
      const icon = darkModeToggle.querySelector('.dark-mode-toggle-icon');
      if (icon) {
        icon.style.display = 'block';
        icon.style.verticalAlign = 'middle';
      }
    }
    
    // Target specifically the blog button for alignment
    if (blogButton) {
      blogButton.style.display = 'flex';
      blogButton.style.alignItems = 'center';
      blogButton.style.verticalAlign = 'middle';
      blogButton.style.height = '100%';
      blogButton.style.paddingTop = '0';
      blogButton.style.paddingBottom = '0';
      blogButton.style.lineHeight = 'normal';
      
      // Also style its parent
      const blogLi = blogButton.closest('li');
      if (blogLi) {
        blogLi.style.display = 'flex';
        blogLi.style.alignItems = 'center';
        blogLi.style.height = '100%';
      }
    }
  }
  
  // Run on page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fixHeaderAlignment);
  } else {
    fixHeaderAlignment();
  }
  
  // Run again after a short delay to ensure everything is loaded
  setTimeout(fixHeaderAlignment, 100);
  
  // Also run on window resize
  window.addEventListener('resize', fixHeaderAlignment);
})();
