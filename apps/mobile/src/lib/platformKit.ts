// Two platform services, split by file so the web bundle carries neither library:
// on web they're one browser call each (platformKit.web.ts).
import * as Crypto from 'expo-crypto';
import * as WebBrowser from 'expo-web-browser';

/** A random v4 UUID (event ids, the dev reader id). */
export function randomUUID(): string {
  return Crypto.randomUUID();
}

/** Opens a publisher's page in the in-app browser. */
export function openUrl(url: string): void {
  void WebBrowser.openBrowserAsync(url);
}
