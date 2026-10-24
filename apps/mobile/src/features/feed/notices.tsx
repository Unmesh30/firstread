// The feed's status lines: offline, busy (retrying after Retry-After), and degraded.
import { useEffect, useState } from 'react';

import { ApiError } from '@/api/errors';
import { MAX_RETRY_AFTER_S } from '@/api/queries';
import { Banner } from '@/ui';

export const DEGRADED_NOTE = 'Some personalization signals are temporarily unavailable';
export const OFFLINE_NOTE =
  'You’re offline. Stories you’ve loaded still work; new ones will load when you’re back.';

/** Seconds left before TanStack retries a 429/503, counting down from Retry-After. */
export function useRetryCountdown(reason: unknown, failureCount: number): number | null {
  const wait =
    reason instanceof ApiError && (reason.status === 429 || reason.status === 503)
      ? Math.min(reason.retryAfterS ?? 1, MAX_RETRY_AFTER_S)
      : null;
  const key = `${wait}:${failureCount}`;
  const [tick, setTick] = useState({ key, elapsed: 0 });
  useEffect(() => {
    if (wait == null) return;
    const started = Date.now();
    const t = setInterval(() => {
      const elapsed = Math.floor((Date.now() - started) / 1000);
      setTick({ key, elapsed });
      if (elapsed >= wait) clearInterval(t);
    }, 1000);
    return () => clearInterval(t);
  }, [key, wait]);
  if (wait == null) return null;
  return Math.max(0, wait - (tick.key === key ? tick.elapsed : 0));
}

export function BusyBanner({ seconds }: { seconds: number }) {
  return (
    <Banner
      tone="warning"
      message={seconds > 0 ? `Busy, retrying in ${seconds} s` : 'Busy, retrying now'}
      testID="busy-banner"
    />
  );
}

export function OfflineBanner() {
  return <Banner tone="offline" message={OFFLINE_NOTE} testID="offline-banner" />;
}

export function DegradedNote() {
  return <Banner tone="info" message={DEGRADED_NOTE} testID="degraded-note" />;
}
