import { fireEvent, render, screen } from '@testing-library/react-native';
import type { ReactElement } from 'react';

import { ApiError } from '@/api/errors';
import { breakpointFor } from '@/hooks/useBreakpoint';
import {
  Banner,
  Button,
  describeError,
  EmptyState,
  ErrorState,
  IconButton,
  ThemeProvider,
} from '@/ui';
import { resolveTheme } from '@/ui/theme';

const themed = (el: ReactElement) => render(<ThemeProvider>{el}</ThemeProvider>);

describe('components', () => {
  it('Button exposes its label, busy and disabled state', async () => {
    const onPress = jest.fn();
    await themed(<Button label="Save" onPress={onPress} />);
    await fireEvent.press(screen.getByRole('button', { name: 'Save' }));
    expect(onPress).toHaveBeenCalledTimes(1);
    await themed(<Button label="Saving" loading onPress={onPress} />);
    expect(screen.getByRole('button', { name: 'Saving' })).toBeDisabled();
  });

  it('IconButton is named by its label and reports selection', async () => {
    await themed(<IconButton icon="heart" label="Like: A headline" selected />);
    expect(screen.getByRole('button', { name: 'Like: A headline' })).toBeSelected();
  });

  it('Banner is a polite status, or an alert for errors', async () => {
    await themed(<Banner tone="offline" message="You’re offline" testID="b1" />);
    expect(screen.getByTestId('b1').props).toMatchObject({ role: 'status', 'aria-live': 'polite' });
    expect(screen.getByTestId('b1')).toHaveTextContent('You’re offline');
    await themed(<Banner tone="danger" message="Broken" testID="b2" />);
    expect(screen.getByTestId('b2').props).toMatchObject({
      role: 'alert',
      'aria-live': 'assertive',
    });
  });

  it('EmptyState and ErrorState explain what happened and offer a way on', async () => {
    const retry = jest.fn();
    await themed(
      <EmptyState
        title="All caught up"
        body="Pull to refresh"
        action={{ label: 'Refresh', onPress: retry }}
      />,
    );
    expect(screen.getByRole('heading', { name: 'All caught up' })).toBeTruthy();
    await themed(
      <ErrorState error={new ApiError(503, 'unavailable', 'x', 'rid-9', 5)} onRetry={retry} />,
    );
    expect(screen.getByText('Request id: rid-9')).toBeTruthy();
    await fireEvent.press(screen.getByRole('button', { name: 'Try again' }));
    expect(retry).toHaveBeenCalledTimes(1);
  });
});

describe('helpers', () => {
  it('describes errors in plain words', () => {
    expect(describeError(new ApiError(0, 'network', ''))).toMatch(/connection/);
    expect(describeError(new ApiError(429, 'rate_limited', '', null, 7))).toMatch(/7 s/);
    expect(describeError(new Error('x'))).toMatch(/our side/);
  });
  it('resolves the theme and breakpoints', () => {
    expect(resolveTheme('system', 'dark')).toBe('dark');
    expect(resolveTheme('system', null)).toBe('light');
    expect(resolveTheme('dark', 'light')).toBe('dark');
    expect([breakpointFor(375), breakpointFor(700), breakpointFor(1280)]).toEqual([
      'compact',
      'medium',
      'expanded',
    ]);
  });
});
