/**
 * Special fix for site name alignment in the header
 */
(function() {
  function fixSiteNameAlignment() {
    // Get the site name link and its container
    const siteNameLink = document.querySelector('.left .nav-button');
    const leftDiv = document.querySelector('nav[role="navigation"] .left');
    const mainNav = document.querySelector('nav[role="navigation"]');
    
    if (!siteNameLink || !leftDiv || !mainNav) return;
    
    // Set natural height for the navigation
    mainNav.style.height = '';
    mainNav.style.minHeight = '';
    
    // Fix the left div containing the site name
    leftDiv.style.display = 'flex';
    leftDiv.style.alignItems = 'center';
    leftDiv.style.height = '';
    leftDiv.style.margin = '0';
    
    // Fix the site name link
    siteNameLink.style.display = 'flex';
    siteNameLink.style.alignItems = 'center';
    siteNameLink.style.justifyContent = 'center';
    siteNameLink.style.height = '';
    siteNameLink.style.paddingTop = '0';
    siteNameLink.style.paddingBottom = '0';
    
    // Get and fix the span containing the text
    const siteNameSpan = siteNameLink.querySelector('span');
    if (siteNameSpan) {
      siteNameSpan.style.display = 'flex';
      siteNameSpan.style.alignItems = 'center';
      siteNameSpan.style.justifyContent = 'center';
      siteNameSpan.style.height = '';
      siteNameSpan.style.lineHeight = 'normal';
    }
  }
  
  // Run on page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fixSiteNameAlignment);
  } else {
    fixSiteNameAlignment();
  }
  
  // Run again after a short delay to ensure everything is loaded
  setTimeout(fixSiteNameAlignment, 100);
  
  // Run again after a longer delay to catch any late changes
  setTimeout(fixSiteNameAlignment, 500);
  
  // Also run on window resize
  window.addEventListener('resize', fixSiteNameAlignment);
})();
