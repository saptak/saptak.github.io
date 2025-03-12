// Minimal fix for hamburger menu
(function() {
  // Remove hamburger toggle
  var toggle = document.querySelector('.nav-toggle');
  if (toggle && toggle.parentNode) {
    toggle.parentNode.removeChild(toggle);
  }
  
  // Force nav to be open
  var nav = document.querySelector('.nav-collapse');
  if (nav) {
    nav.classList.remove('closed');
    nav.classList.add('opened');
    nav.removeAttribute('aria-hidden');
  }
})();
