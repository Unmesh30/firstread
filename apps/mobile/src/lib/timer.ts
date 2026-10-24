// A countdown that can be paused for several reasons at once (hover, focus, hidden
// page) and resumes with the time that was left (WCAG 2.2.1 Timing Adjustable).

export type PauseReason = 'hover' | 'focus' | 'hidden';

export class PausableTimer {
  private remainingMs: number;
  private startedAt: number | null = null;
  private handle: ReturnType<typeof setTimeout> | null = null;
  private readonly reasons = new Set<PauseReason>();
  private done = false;

  constructor(
    durationMs: number,
    private readonly onDone: () => void,
    private readonly now: () => number = Date.now,
  ) {
    this.remainingMs = durationMs;
  }

  start(): void {
    if (this.done || this.handle || this.reasons.size) return;
    this.startedAt = this.now();
    this.handle = setTimeout(() => this.finish(), this.remainingMs);
  }

  pause(reason: PauseReason): void {
    this.reasons.add(reason);
    if (!this.handle || this.startedAt == null) return;
    clearTimeout(this.handle);
    this.handle = null;
    this.remainingMs = Math.max(0, this.remainingMs - (this.now() - this.startedAt));
    this.startedAt = null;
  }

  resume(reason: PauseReason): void {
    this.reasons.delete(reason);
    if (!this.reasons.size) this.start();
  }

  /** Finish now (e.g. the page is closing): runs onDone once. */
  finish(): void {
    if (this.done) return;
    this.done = true;
    if (this.handle) clearTimeout(this.handle);
    this.handle = null;
    this.onDone();
  }

  /** Stop without running onDone (e.g. Undo was pressed). */
  cancel(): void {
    this.done = true;
    if (this.handle) clearTimeout(this.handle);
    this.handle = null;
  }

  get paused(): boolean {
    return this.reasons.size > 0;
  }

  get remaining(): number {
    if (this.startedAt == null) return this.remainingMs;
    return Math.max(0, this.remainingMs - (this.now() - this.startedAt));
  }
}
