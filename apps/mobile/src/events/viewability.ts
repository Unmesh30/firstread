// When a card counts as seen: ≥ 50% of it on screen for ≥ 1 continuous second while
// the page or app is in the foreground (the MRC display viewability rule), and once per
// (feed request, article).
//
// FlatList reports which rows cross the 50% line (`itemVisiblePercentThreshold`) as soon
// as that changes; this tracker owns the clock. FlatList's own `minimumViewTime` isn't
// used: its timer only re-checks that a row is visible when it fires, so a row that
// leaves and comes back within the second still counts, and it keeps running in a
// background tab. Here leaving cancels the clock, and the background pauses it.

export const VIEW_PERCENT = 50;
export const VIEW_MS = 1000;

/** Pass to FlatList: rows are reported the moment they cross 50%, not after a delay. */
export const viewabilityConfig = { itemVisiblePercentThreshold: VIEW_PERCENT };

export class ViewabilityTracker<T> {
  private visible = new Map<string, T>();
  private readonly timers = new Map<string, ReturnType<typeof setTimeout>>();
  private readonly counted = new Set<string>();
  private foreground = true;
  private onSeen: (key: string, item: T) => void;

  constructor(
    onSeen: (key: string, item: T) => void = () => undefined,
    private readonly ms: number = VIEW_MS,
  ) {
    this.onSeen = onSeen;
  }

  /** Where counted impressions go (the event queue, once there is one). */
  setSink(onSeen: (key: string, item: T) => void): void {
    this.onSeen = onSeen;
  }

  /** The rows that are ≥ 50% visible now (all of them, not a diff). */
  update(rows: Iterable<[key: string, item: T]>): void {
    const next = new Map(rows);
    for (const key of this.visible.keys()) if (!next.has(key)) this.clear(key);
    this.visible = next;
    this.startAll();
  }

  setForeground(foreground: boolean): void {
    if (this.foreground === foreground) return;
    this.foreground = foreground;
    if (foreground) this.startAll();
    else for (const key of [...this.timers.keys()]) this.clear(key);
  }

  hasCounted(key: string): boolean {
    return this.counted.has(key);
  }

  dispose(): void {
    for (const key of [...this.timers.keys()]) this.clear(key);
    this.visible.clear();
  }

  private startAll(): void {
    if (!this.foreground) return;
    for (const [key, item] of this.visible) {
      if (this.counted.has(key) || this.timers.has(key)) continue;
      this.timers.set(
        key,
        setTimeout(() => {
          this.timers.delete(key);
          this.counted.add(key);
          this.onSeen(key, item);
        }, this.ms),
      );
    }
  }

  private clear(key: string): void {
    const t = this.timers.get(key);
    if (t) clearTimeout(t);
    this.timers.delete(key);
  }
}
