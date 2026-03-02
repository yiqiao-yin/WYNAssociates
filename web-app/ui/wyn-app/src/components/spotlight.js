export function triggerSpotlight(target, duration = 4000) {
  const el = document.querySelector(`[data-spotlight="${target}"]`);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  el.classList.add('spotlight-active');
  setTimeout(() => {
    el.classList.remove('spotlight-active');
  }, duration);
}
