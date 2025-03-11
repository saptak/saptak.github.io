/**
 * Debug script to check what's going on with the dark mode toggle button
 */
(function() {
  console.log('Running dark mode button debug script');
  
  // Check if the button exists
  const button = document.getElementById('dark-mode-toggle');
  if (!button) {
    console.log('Dark mode toggle button not found!');
    return;
  }
  
  // Log the button's HTML content
  console.log('Button HTML:', button.innerHTML);
  
  // Log the button's computed style
  const buttonStyle = window.getComputedStyle(button);
  console.log('Button padding:', buttonStyle.padding);
  console.log('Button margin:', buttonStyle.margin);
  console.log('Button width:', buttonStyle.width);
  console.log('Button height:', buttonStyle.height);
  
  // Check if there's a span inside
  const spans = button.querySelectorAll('span');
  console.log('Spans found inside button:', spans.length);
  spans.forEach((span, index) => {
    console.log(`Span ${index} text:`, span.textContent);
    console.log(`Span ${index} display:`, window.getComputedStyle(span).display);
  });
  
  // Check if there's any text node inside the button
  let hasTextNode = false;
  button.childNodes.forEach(node => {
    if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
      console.log('Found text node with content:', node.textContent);
      hasTextNode = true;
    }
  });
  if (!hasTextNode) {
    console.log('No text nodes found with content');
  }
  
  // Add a click handler to debug the toggle function
  button.addEventListener('click', function() {
    console.log('Button clicked');
    console.log('Current theme:', document.documentElement.getAttribute('data-theme'));
    setTimeout(() => {
      console.log('Theme after click:', document.documentElement.getAttribute('data-theme'));
    }, 100);
  });
})();
