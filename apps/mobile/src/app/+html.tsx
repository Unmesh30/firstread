// The HTML shell for every statically rendered page (web only; runs in Node at build
// time, so nothing here may touch browser APIs).
import { ScrollViewStyleReset, useServerDocumentContext } from 'expo-router/html';
import type { PropsWithChildren } from 'react';

import { env } from '@/lib/env';
import { OG_IMAGE, SITE_DESCRIPTION, SITE_TITLE, THEME_SCRIPT } from '@/lib/site';
import { responsiveCss } from '@/ui/responsive';
import { colors, colorVarDeclarations } from '@/ui/tokens';

// Paint the right theme before any JavaScript runs, so a dark-mode reader never sees
// light surfaces: until hydration the components are styled with CSS variables
// (tokens.colorVars), defined here for the system scheme, or for the reader's stored
// choice (data-theme, set by the script below; the theme provider keeps it in step).
const themeCss = `
:root { ${colorVarDeclarations('light')} color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { ${colorVarDeclarations('dark')} color-scheme: dark; }
}
:root[data-theme="dark"] { ${colorVarDeclarations('dark')} color-scheme: dark; }
html, body { background-color: var(--fr-background); color: var(--fr-text); }
${responsiveCss}`;

export default function Root({ children }: PropsWithChildren) {
  const { htmlAttributes, bodyAttributes, headNodes, bodyNodes } = useServerDocumentContext();
  return (
    <html lang="en" {...htmlAttributes}>
      <head>
        <meta charSet="utf-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta name="description" content={SITE_DESCRIPTION} />
        <meta
          name="theme-color"
          media="(prefers-color-scheme: light)"
          content={colors.light.background}
        />
        <meta
          name="theme-color"
          media="(prefers-color-scheme: dark)"
          content={colors.dark.background}
        />
        {/* Share cards (Open Graph, and X/Twitter's large-image card). */}
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="FirstRead" />
        <meta property="og:title" content={SITE_TITLE} />
        <meta property="og:description" content={SITE_DESCRIPTION} />
        <meta property="og:url" content={`${env.siteUrl}/`} />
        <meta property="og:image" content={OG_IMAGE.url} />
        <meta property="og:image:type" content="image/png" />
        <meta property="og:image:width" content={String(OG_IMAGE.width)} />
        <meta property="og:image:height" content={String(OG_IMAGE.height)} />
        <meta property="og:image:alt" content={OG_IMAGE.alt} />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={SITE_TITLE} />
        <meta name="twitter:description" content={SITE_DESCRIPTION} />
        <meta name="twitter:image" content={OG_IMAGE.url} />
        <meta name="twitter:image:alt" content={OG_IMAGE.alt} />
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
        <ScrollViewStyleReset />
        <style dangerouslySetInnerHTML={{ __html: themeCss }} />
        {headNodes}
      </head>
      <body {...bodyAttributes}>
        {children}
        {bodyNodes}
      </body>
    </html>
  );
}
