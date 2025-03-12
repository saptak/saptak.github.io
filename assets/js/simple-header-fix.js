/**
 * Simple header fix that addresses all issues at once
 */
(function() {
  // Remove the hamburger toggle
  const toggles = document.querySelectorAll('.nav-toggle');
  toggles.forEach(function(toggle) {
    if (toggle && toggle.parentNode) {
      toggle.parentNode.removeChild(toggle);
    }
  });
  
  // Make navigation always visible
  const navs = document.querySelectorAll('.nav-collapse');
  navs.forEach(function(nav) {
    nav.classList.remove('closed');
    nav.classList.add('opened');
    nav.removeAttribute('aria-hidden');
  });
})();
