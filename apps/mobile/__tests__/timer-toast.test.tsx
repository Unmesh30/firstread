import { act, fireEvent, render, screen } from '@testing-library/react-native';

import { PausableTimer } from '@/lib/timer';
import { ThemeProvider, ToastHost, useToasts } from '@/ui';

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
