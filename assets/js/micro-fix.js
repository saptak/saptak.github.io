// Ultra-minimal fix for hamburger menu
(function() {
  // Remove the hamburger toggle from DOM
  var toggle = document.querySelector('.nav-toggle');
  if (toggle && toggle.parentNode) {
    toggle.parentNode.removeChild(toggle);
  }
})();
