// The HTML shell for every statically rendered page (web only; runs in Node at build
// time, so nothing here may touch browser APIs).
import { ScrollViewStyleReset, useServerDocumentContext } from 'expo-router/html';
import type { PropsWithChildren } from 'react';

import { responsiveCss } from '@/ui/responsive';
import { colors } from '@/ui/tokens';

// Paint the right background before any JavaScript runs, so dark mode never flashes white.
const preload = `
html, body { background-color: ${colors.light.background}; color: ${colors.light.text}; }
@media (prefers-color-scheme: dark) {
  html, body { background-color: ${colors.dark.background}; color: ${colors.dark.text}; }
}
:root { color-scheme: light dark; }
${responsiveCss}`;

export default function Root({ children }: PropsWithChildren) {
  const { htmlAttributes, bodyAttributes, headNodes, bodyNodes } = useServerDocumentContext();
  return (
    <html lang="en" {...htmlAttributes}>
      <head>
        <meta charSet="utf-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta
          name="description"
          content="Breaking news, ranked for you before anyone has clicked."
        />
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
        <ScrollViewStyleReset />
        <style dangerouslySetInnerHTML={{ __html: preload }} />
        {headNodes}
      </head>
      <body {...bodyAttributes}>
        {children}
        {bodyNodes}
      </body>
    </html>
  );
}
