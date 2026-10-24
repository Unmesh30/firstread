import { render, screen } from '@testing-library/react-native';
import { renderToString } from 'react-dom/server';
import { Dimensions, Text } from 'react-native';

import { breakpointFor } from '@/hooks/useBreakpoint';
import { SERVER_WIDTH, useHydrated, useSystemScheme, useWindowWidth } from '@/hooks/useHydrated';
import { resolveTheme } from '@/ui/theme';

function Probe() {
  return <>{`${useHydrated()} ${useWindowWidth()} ${useSystemScheme() ?? 'none'}`}</>;
}

describe('hydration-safe hooks', () => {
  test('the static render sees no window: not hydrated, the server width, no scheme', () => {
    expect(renderToString(<Probe />)).toBe(`false ${SERVER_WIDTH} none`);
    // …so static pages are the compact layout in the light theme.
    expect(breakpointFor(SERVER_WIDTH)).toBe('compact');
    expect(resolveTheme('system', null)).toBe('light');
  });

  test('a client render (native, or after hydration) sees the real values', async () => {
    const { width } = Dimensions.get('window');
    await render(
      <Text testID="probe">
        <Probe />
      </Text>,
    );
    expect(screen.getByTestId('probe')).toHaveTextContent(new RegExp(`^true ${width} `));
  });
});
