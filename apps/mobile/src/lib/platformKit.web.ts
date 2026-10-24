// The web side of platformKit.ts: what expo-crypto and expo-web-browser would do in a
// browser, without bundling them (4.8 KB gzip). Links open their own tabs on web, so
// openUrl is only a fallback.

/** A random v4 UUID (the Web Crypto API; GitHub Pages and localhost are secure contexts). */
export function randomUUID(): string {
  return globalThis.crypto.randomUUID();
}

export function openUrl(url: string): void {
  window.open(url, '_blank', 'noopener,noreferrer');
}
