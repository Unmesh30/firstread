// What the HTML shell (app/+html.tsx) says about the site: the share-card tags and the
// pre-paint theme script. Plain constants, safe in Node at build time.
import { env } from './env';

export const SITE_TITLE = 'FirstRead: breaking news, ranked before the clicks';
export const SITE_DESCRIPTION = 'Breaking news, ranked for you before anyone has clicked.';

/** public/og-image.png (scripts/make-og-image.mjs); share cards need an absolute URL. */
export const OG_IMAGE = {
  url: `${env.siteUrl}/og-image.png`,
  width: 1200,
  height: 630,
  alt: 'FirstRead: a For You news feed that ranks brand-new stories before they have clicks.',
};

/** Where the theme provider stores the reader's choice on web (kv prefix "fr:" + "theme"). */
export const THEME_STORAGE_KEY = 'fr:theme';

/** Applies a stored Light/Dark choice before the first paint (an inline <head> script). */
export const THEME_SCRIPT = `try{var t=localStorage.getItem('${THEME_STORAGE_KEY}');if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)}catch(e){}`;
