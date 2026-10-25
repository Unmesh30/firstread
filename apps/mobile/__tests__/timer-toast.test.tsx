import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen } from '@testing-library/react-native';

import { useArticleActions } from '@/features/feed/actions';
import type { FeedRow } from '@/features/feed/useFeed';
import { focusWhenPresent } from '@/lib/focus';
import { PausableTimer } from '@/lib/timer';
import { ThemeProvider, ToastHost, useToasts } from '@/ui';
import { hideButtonId } from '@/ui/ArticleCard';

beforeEach(() => {
  jest.useFakeTimers();
  useToasts.setState({ toasts: [] });
});
afterEach(() => jest.useRealTimers());

describe('PausableTimer', () => {
  it('pauses for several reasons and resumes with the time that was left', () => {
    let t = 0;
    const done = jest.fn();
    const timer = new PausableTimer(5000, done, () => t);
    timer.start();
    t = 2000;
    jest.advanceTimersByTime(2000);
    timer.pause('hover');
    timer.pause('focus');
    expect(timer.remaining).toBe(3000);
    t = 60_000;
    jest.advanceTimersByTime(58_000); // paused: nothing happens
    timer.resume('hover');
    expect(timer.paused).toBe(true); // still focused
    jest.advanceTimersByTime(10_000);
    expect(done).not.toHaveBeenCalled();
    timer.resume('focus');
    jest.advanceTimersByTime(2999);
    expect(done).not.toHaveBeenCalled();
    jest.advanceTimersByTime(1);
    expect(done).toHaveBeenCalledTimes(1);
  });
  it('finishes once, and a cancelled timer never fires', () => {
    const done = jest.fn();
    const a = new PausableTimer(1000, done);
    a.start();
    a.finish();
    a.finish();
    jest.advanceTimersByTime(5000);
    expect(done).toHaveBeenCalledTimes(1);
    const cancelled = jest.fn();
    const b = new PausableTimer(1000, cancelled);
    b.start();
    b.cancel();
    jest.advanceTimersByTime(5000);
    expect(cancelled).not.toHaveBeenCalled();
  });
});

function renderHost() {
  return render(
    <ThemeProvider>
      <ToastHost />
    </ThemeProvider>,
  );
}

describe('undo toast', () => {
  it('expires after 5 s unless Undo is pressed', async () => {
    await renderHost();
    const expire = jest.fn();
    const undo = jest.fn();
    await act(async () => {
      useToasts
        .getState()
        .show({ message: 'Story hidden', actionLabel: 'Undo', onAction: undo, onExpire: expire });
    });
    expect(screen.getByText('Story hidden. Undo available')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Undo' })).toBeTruthy();
    await act(async () => {
      jest.advanceTimersByTime(4999);
    });
    expect(expire).not.toHaveBeenCalled();
    await act(async () => {
      jest.advanceTimersByTime(1);
    });
    expect(expire).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId('toast')).toBeNull();

    await act(async () => {
      useToasts
        .getState()
        .show({ message: 'Again', actionLabel: 'Undo', onAction: undo, onExpire: expire });
    });
    await fireEvent.press(screen.getByRole('button', { name: 'Undo' }));
    await act(async () => {
      jest.advanceTimersByTime(10_000);
    });
    expect(undo).toHaveBeenCalledTimes(1);
    expect(expire).toHaveBeenCalledTimes(1);
  });

  it('pauses while the Undo button has focus', async () => {
    await renderHost();
    const expire = jest.fn();
    await act(async () => {
      useToasts.getState().show({ message: 'Story hidden', actionLabel: 'Undo', onExpire: expire });
    });
    await fireEvent(screen.getByRole('button', { name: 'Undo' }), 'focus');
    await act(async () => {
      jest.advanceTimersByTime(30_000);
    });
    expect(expire).not.toHaveBeenCalled();
    await fireEvent(screen.getByRole('button', { name: 'Undo' }), 'blur');
    await act(async () => {
      jest.advanceTimersByTime(5000);
    });
    expect(expire).toHaveBeenCalledTimes(1);
  });

  it('commits a pending hide when the page is hidden, and pauses other toasts', async () => {
    const commit = jest.fn();
    const other = jest.fn();
    await act(async () => {
      useToasts.getState().show({
        message: 'Story hidden',
        actionLabel: 'Undo',
        onExpire: commit,
        commitOnHide: true,
      });
      useToasts.getState().show({ message: 'Feed refreshed', onExpire: other });
    });
    await act(async () => {
      useToasts.getState().hidden();
    });
    expect(commit).toHaveBeenCalledTimes(1);
    await act(async () => {
      jest.advanceTimersByTime(60_000);
    });
    expect(other).not.toHaveBeenCalled(); // paused while hidden
    await act(async () => {
      useToasts.getState().visible();
    });
    await act(async () => {
      jest.advanceTimersByTime(5000);
    });
    expect(other).toHaveBeenCalledTimes(1);
  });
});

describe('focus after a keyboard Undo', () => {
  function fakeDom(appearsOnFrame: number | null) {
    const frames: (() => void)[] = [];
    let frame = 0;
    const el = { focus: jest.fn() };
    return {
      el,
      frames,
      run() {
        while (frames.length) {
          frame++;
          frames.shift()!();
        }
        return frame;
      },
      deps: {
        getElementById: (id: string) =>
          id === 'hide-button-a1' && appearsOnFrame != null && frame >= appearsOnFrame ? el : null,
        requestAnimationFrame: (cb: () => void) => frames.push(cb),
        cancelAnimationFrame: jest.fn(),
      },
    };
  }

  it('focuses the restored card’s hide button once it is back in the DOM', () => {
    const dom = fakeDom(3);
    focusWhenPresent(hideButtonId('a1'), dom.deps);
    expect(dom.run()).toBe(3);
    expect(dom.el.focus).toHaveBeenCalledTimes(1);
  });

  it('gives up after about half a second if the card never returns', () => {
    const dom = fakeDom(null);
    focusWhenPresent(hideButtonId('a1'), dom.deps);
    expect(dom.run()).toBe(30);
    expect(dom.el.focus).not.toHaveBeenCalled();
  });

  it('a hide’s undo toast returns focus to that card’s hide button (story or source)', async () => {
    const row = {
      item: { id: 'a1', source_name: 'Example Daily', source_domain: 'daily.example.com' },
      requestId: 'req-1',
      position: 0,
    } as unknown as FeedRow;
    let hide: ReturnType<typeof useArticleActions>['hide'] = () => 0;
    function Probe() {
      hide = useArticleActions().hide;
      return null;
    }
    await render(
      <QueryClientProvider client={new QueryClient()}>
        <Probe />
      </QueryClientProvider>,
    );
    for (const scope of ['article', 'source'] as const) {
      await act(async () => {
        hide(row, scope);
      });
      expect(useToasts.getState().toasts.at(-1)?.returnFocusTo).toBe('hide-button-a1');
      useToasts.getState().discardAll();
    }
  });
});
