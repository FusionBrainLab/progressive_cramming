/* Light interactivity for the Progressive Cramming deck.
   Auto-play the trajectory <video> on the slide you land on; pause/rewind the rest.
   Pure vanilla JS; no network. */
(function () {
  function syncVideos(currentSlide) {
    document.querySelectorAll('.slides video').forEach(function (v) {
      if (currentSlide && currentSlide.contains(v)) {
        try { v.currentTime = 0; v.play(); } catch (e) {}
      } else {
        v.pause();
      }
    });
  }

  function init() {
    if (window.Reveal) {
      Reveal.on('slidechanged', function (e) { syncVideos(e.currentSlide); });
      Reveal.on('ready', function (e) { syncVideos(e.currentSlide); });
    }
  }
  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
