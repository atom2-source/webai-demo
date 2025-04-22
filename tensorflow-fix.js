// This script helps TensorFlow.js find models correctly when hosted on GitHub Pages
if (window.location.hostname.includes('github.io')) {
  console.log('Running on GitHub Pages, applying TensorFlow.js model paths fix');
  
  // Set up a global path prefix for TensorFlow.js models
  window.tfModelPathPrefix = window.location.pathname.endsWith('/') 
    ? window.location.pathname 
    : window.location.pathname + '/';
    
  // Intercept fetch requests to help with model loading
  const originalFetch = window.fetch;
  window.fetch = function(url, options) {
    if (typeof url === 'string' && url.includes('model.json') && !url.startsWith('http')) {
      // Convert relative URL to absolute URL if needed
      if (url.startsWith('./')) {
        url = window.tfModelPathPrefix + url.substring(2);
      } else if (!url.startsWith('/')) {
        url = window.tfModelPathPrefix + url;
      }
      console.log('TensorFlow.js model URL adjusted:', url);
    }
    return originalFetch(url, options);
  };
}
