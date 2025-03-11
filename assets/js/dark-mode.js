/**
 * Dark mode functionality
 */
(function() {
  // Check for saved theme preference or use the system preference
  const savedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  // Set the theme on page load
  if (savedTheme === 'dark' || (savedTheme === null && prefersDark)) {
    document.documentElement.setAttribute('data-theme', 'dark');
    updateToggleButton(true);
  } else {
    document.documentElement.setAttribute('data-theme', 'light');
    updateToggleButton(false);
  }
  
  // Toggle the theme when the button is clicked
  document.addEventListener('DOMContentLoaded', function() {
    const toggleButton = document.getElementById('dark-mode-toggle');
    
    if (toggleButton) {
      toggleButton.addEventListener('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        updateToggleButton(newTheme === 'dark');
      });
    }
  });
  
  // Listen for system preference changes
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
      // Only change theme automatically if user hasn't manually set a preference
      if (!localStorage.getItem('theme')) {
        const newTheme = e.matches ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        updateToggleButton(e.matches);
      }
    });
  }
  
  // Update the toggle button appearance
  function updateToggleButton(isDark) {
    document.addEventListener('DOMContentLoaded', function() {
      const toggleButton = document.getElementById('dark-mode-toggle');
      if (!toggleButton) return;
      
      const toggleText = toggleButton.querySelector('.toggle-text');
      const sunIcon = toggleButton.querySelector('.sun-icon');
      const moonIcon = toggleButton.querySelector('.moon-icon');
      
      if (toggleText) toggleText.textContent = isDark ? 'Light Mode' : 'Dark Mode';
      
      if (sunIcon && moonIcon) {
        if (isDark) {
          sunIcon.style.display = 'none';
          moonIcon.style.display = 'block';
        } else {
          sunIcon.style.display = 'block';
          moonIcon.style.display = 'none';
        }
      }
    });
  }
})();