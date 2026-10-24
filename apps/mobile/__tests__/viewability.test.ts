import { VIEW_MS, ViewabilityTracker } from '@/events/viewability';

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

function setup() {
  const seen: string[] = [];
  const t = new ViewabilityTracker<string>((key) => seen.push(key));
  return { t, seen };
}

describe('viewability: ≥ 50% for ≥ 1 continuous second in the foreground', () => {
  it('counts a row after a full second on screen', () => {
    const { t, seen } = setup();
    t.update([['r1:a', 'a']]);
    jest.advanceTimersByTime(VIEW_MS - 1);
    expect(seen).toEqual([]);
    jest.advanceTimersByTime(1);
    expect(seen).toEqual(['r1:a']);
  });

  it('a row that leaves before the second is up starts over when it returns', () => {
    const { t, seen } = setup();
    t.update([['r1:a', 'a']]);
    jest.advanceTimersByTime(600);
    t.update([]); // scrolled away
    jest.advanceTimersByTime(300);
    t.update([['r1:a', 'a']]); // back
    jest.advanceTimersByTime(700); // 600 + 700 > 1 s, but not continuous
    expect(seen).toEqual([]);
    jest.advanceTimersByTime(300);
    expect(seen).toEqual(['r1:a']);
  });

  it('counts once per (request, article)', () => {
    const { t, seen } = setup();
    t.update([['r1:a', 'a']]);
    jest.advanceTimersByTime(VIEW_MS);
    t.update([]);
    t.update([['r1:a', 'a']]);
    jest.advanceTimersByTime(5 * VIEW_MS);
    expect(seen).toEqual(['r1:a']);
    expect(t.hasCounted('r1:a')).toBe(true);
    // The same article served by a new first page is a new impression.
    t.update([['r2:a', 'a']]);
    jest.advanceTimersByTime(VIEW_MS);
    expect(seen).toEqual(['r1:a', 'r2:a']);
  });

  it('pauses in the background and needs a fresh second after', () => {
    const { t, seen } = setup();
    t.update([
      ['r1:a', 'a'],
      ['r1:b', 'b'],
    ]);
    jest.advanceTimersByTime(900);
    t.setForeground(false);
    jest.advanceTimersByTime(60_000);
    expect(seen).toEqual([]);
    t.setForeground(true);
    jest.advanceTimersByTime(VIEW_MS - 1);
    expect(seen).toEqual([]);
    jest.advanceTimersByTime(1);
    expect(seen.sort()).toEqual(['r1:a', 'r1:b']);
  });

  it('rows that become visible in the background wait for the foreground', () => {
    const { t, seen } = setup();
    t.setForeground(false);
    t.update([['r1:a', 'a']]);
    jest.advanceTimersByTime(5000);
    expect(seen).toEqual([]);
    t.setForeground(true);
    jest.advanceTimersByTime(VIEW_MS);
    expect(seen).toEqual(['r1:a']);
  });

  it('sends counted rows to the current sink', () => {
    const t = new ViewabilityTracker<string>();
    const sink = jest.fn();
    t.setSink(sink);
    t.update([['r1:a', 'a']]);
    jest.advanceTimersByTime(VIEW_MS);
    expect(sink).toHaveBeenCalledWith('r1:a', 'a');
    t.dispose();
  });
});
