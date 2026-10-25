// Moving keyboard focus to an element that may not be in the DOM yet (web only): a
// restored card renders a frame or two after the state change that brings it back.

const MAX_FRAMES = 30; // ~0.5 s at 60 fps

interface FocusDeps {
  getElementById(id: string): { focus(): void } | null;
  requestAnimationFrame(cb: () => void): number;
  cancelAnimationFrame(handle: number): void;
}

function browser(): FocusDeps {
  return {
    getElementById: (id) => document.getElementById(id),
    requestAnimationFrame: (cb) => window.requestAnimationFrame(cb),
    cancelAnimationFrame: (h) => window.cancelAnimationFrame(h),
  };
}

/** Focuses `#id` as soon as it exists, checking once per frame; gives up after ~0.5 s. Returns a cancel. */
export function focusWhenPresent(id: string, deps: FocusDeps = browser()): () => void {
  let frames = 0;
  let handle = 0;
  const attempt = () => {
    const el = deps.getElementById(id);
    if (el) {
      el.focus();
      return;
    }
    if (++frames < MAX_FRAMES) handle = deps.requestAnimationFrame(attempt);
  };
  handle = deps.requestAnimationFrame(attempt);
  return () => deps.cancelAnimationFrame(handle);
}
