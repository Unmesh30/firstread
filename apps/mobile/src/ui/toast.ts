// Toasts: one visible at a time, each with an optional action (e.g. Undo).
//
// `onExpire` runs when the toast closes on its own; `onAction` instead of it when the
// action is pressed. A toast with `commitOnHide` (a pending hide) is expired at once
// when the page or app goes to the background, so the event queue's flush on 'hidden'
// carries it (call `commitPending()` before flushing).
import { create } from 'zustand';

import { PausableTimer, type PauseReason } from '@/lib/timer';

export const TOAST_MS = 5000;

export interface ToastSpec {
  message: string;
  actionLabel?: string;
  onAction?: () => void;
  onExpire?: () => void;
  durationMs?: number;
  commitOnHide?: boolean;
}

export interface ToastItem extends ToastSpec {
  id: number;
  timer: PausableTimer;
  /** Move keyboard focus to the action (ToastHost does it; focus also pauses the timer). */
  focusAction?: boolean;
}

interface ToastState {
  toasts: ToastItem[];
  show(spec: ToastSpec): number;
  act(id: number): void;
  dismiss(id: number): void;
  pause(id: number, reason: PauseReason): void;
  resume(id: number, reason: PauseReason): void;
  /** Page hidden: commit pending toasts that ask for it, pause the rest. */
  hidden(): void;
  visible(): void;
  commitPending(): void;
  /** Focus this toast's action, if it's still showing (a keyboard reader's Undo). */
  focusAction(id: number): void;
  /** Close every toast without its onExpire (Delete my data: nothing may be committed). */
  discardAll(): void;
}

let nextId = 1;

export const useToasts = create<ToastState>((set, get) => {
  const remove = (id: number) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
  const find = (id: number) => get().toasts.find((t) => t.id === id);
  return {
    toasts: [],
    show(spec) {
      const id = nextId++;
      const timer = new PausableTimer(spec.durationMs ?? TOAST_MS, () => {
        remove(id);
        spec.onExpire?.();
      });
      set((s) => ({ toasts: [...s.toasts, { ...spec, id, timer }] }));
      timer.start();
      return id;
    },
    act(id) {
      const t = find(id);
      if (!t) return;
      t.timer.cancel();
      remove(id);
      t.onAction?.();
    },
    dismiss(id) {
      find(id)?.timer.finish(); // closing early counts as expiring (the hide is kept)
    },
    pause(id, reason) {
      find(id)?.timer.pause(reason);
    },
    resume(id, reason) {
      find(id)?.timer.resume(reason);
    },
    focusAction(id) {
      set((s) => ({
        toasts: s.toasts.map((t) => (t.id === id ? { ...t, focusAction: true } : t)),
      }));
    },
    commitPending() {
      for (const t of get().toasts) if (t.commitOnHide) t.timer.finish();
    },
    discardAll() {
      for (const t of get().toasts) t.timer.cancel();
      set({ toasts: [] });
    },
    hidden() {
      get().commitPending();
      for (const t of get().toasts) t.timer.pause('hidden');
    },
    visible() {
      for (const t of get().toasts) t.timer.resume('hidden');
    },
  };
});
