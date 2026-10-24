// A link to another site. On web it's a real <a target="_blank" rel="noopener
// noreferrer"> (middle-click and screen readers work); on native it opens the in-app
// browser. The accessible name says it opens a new tab.
import type { ReactNode } from 'react';
import { Platform } from 'react-native';

import { openUrl } from '@/lib/platformKit';

import { Text, type Tone } from './Text';
import type { TypeVariant } from './tokens';

export const NEW_TAB = '(opens in a new tab)';

export interface ExternalLinkProps {
  href: string;
  children: ReactNode;
  /** The visible text, when `children` isn't a plain string (for the accessible name). */
  label?: string;
  variant?: TypeVariant;
  tone?: Tone;
  testID?: string;
}

export function ExternalLink({
  href,
  children,
  label,
  variant = 'body',
  tone = 'accent',
  testID,
}: ExternalLinkProps) {
  const name = `${label ?? (typeof children === 'string' ? children : href)} ${NEW_TAB}`;
  const props = Platform.select({
    web: { href, hrefAttrs: { target: '_blank', rel: 'noopener noreferrer' } },
    default: { onPress: () => openUrl(href) },
  }) as object;
  return (
    <Text
      {...props}
      role="link"
      accessibilityLabel={name}
      variant={variant}
      tone={tone}
      weight="600"
      style={{ textDecorationLine: 'underline' }}
      testID={testID}
    >
      {children}
    </Text>
  );
}
