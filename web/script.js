// Copy-to-clipboard
document.querySelectorAll('.copy-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    var text = this.dataset.copy;
    var self = this;
    var original = self.textContent;

    navigator.clipboard.writeText(text).then(function() {
      self.textContent = 'Copied!';
      self.classList.add('copied');
      setTimeout(function() {
        self.textContent = original;
        self.classList.remove('copied');
      }, 2000);
    }).catch(function() {
      // fallback: select text in a temp input
      var el = document.createElement('textarea');
      el.value = text;
      el.style.position = 'fixed';
      el.style.opacity = '0';
      document.body.appendChild(el);
      el.focus();
      el.select();
      try {
        document.execCommand('copy');
        self.textContent = 'Copied!';
        self.classList.add('copied');
        setTimeout(function() {
          self.textContent = original;
          self.classList.remove('copied');
        }, 2000);
      } catch (e) {}
      document.body.removeChild(el);
    });
  });
});

// Nav scroll border
var nav = document.getElementById('nav');
window.addEventListener('scroll', function() {
  nav.classList.toggle('scrolled', window.scrollY > 0);
}, { passive: true });
