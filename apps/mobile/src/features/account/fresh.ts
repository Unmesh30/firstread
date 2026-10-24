// Starting over as a new reader on this device (after Delete my data, or "Reset demo
// identity"). Order matters: nothing that is still waiting may reach the server for
// the old reader, and the new identity must be in place before any query refetches
// (a request as the old reader would re-create an empty profile for it).
import type { QueryClient } from '@tanstack/react-query';
import { router } from 'expo-router';

import type { AuthProvider } from '@/auth';
import type { EventQueue } from '@/events/queue';
import { useFeedLocal } from '@/features/feed/store';
import { useToasts } from '@/ui/toast';

/** Drop what hasn't been sent: hides inside their undo window, queued events. */
export function discardPending(queue: EventQueue | null) {
  useToasts.getState().discardAll();
  queue?.clear();
}

export async function startFresh(auth: AuthProvider, qc: QueryClient, message: string) {
  await auth.reset();
  qc.clear();
  useFeedLocal.getState().reset();
  useToasts.getState().show({ message });
  router.replace('/'); // the gate: onboarding for the new reader
}
