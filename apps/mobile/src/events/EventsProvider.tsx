// One event queue per signed-in reader (stored under their id), wired to the page
// lifecycle and the network: hidden → commit pending hides, one keepalive batch;
// back online → flush.
import { onlineManager } from '@tanstack/react-query';
import { createContext, useContext, useEffect, useMemo, type PropsWithChildren } from 'react';

import { subscribeLifecycle } from '@/lib/lifecycle';
import { randomUUID } from '@/lib/platformKit';
import { useSession } from '@/providers/session';
import { kv, type KV } from '@/storage/kv';
import { useToasts } from '@/ui/toast';

import { EventQueue, type Transport } from './queue';
import { httpTransport } from './transport';

const QueueContext = createContext<EventQueue | null>(null);

export function EventsProvider({
  children,
  store = kv,
  transport,
}: PropsWithChildren<{ store?: KV; transport?: Transport }>) {
  const session = useSession();
  const ready = session.status === 'ready' ? session : null;
  const userId = ready?.session.userId;
  const auth = ready?.auth;
  const apiUrl = ready?.apiUrl;

  const queue = useMemo(
    () =>
      userId && auth && apiUrl
        ? new EventQueue({
            transport: transport ?? httpTransport(apiUrl, auth),
            store,
            storageKey: `events:${userId}`,
            newId: randomUUID,
            isOnline: () => onlineManager.isOnline(),
          })
        : null,
    [userId, auth, apiUrl, store, transport],
  );

  useEffect(() => {
    const q = queue;
    if (!q) return;
    q.start();
    void q.load();
    const offLifecycle = subscribeLifecycle((state) =>
      state === 'hidden'
        ? q.pageHidden(() => useToasts.getState().commitPending())
        : q.pageVisible(),
    );
    const offOnline = onlineManager.subscribe((online) => {
      if (online) void q.flush();
    });
    return () => {
      offLifecycle();
      offOnline();
      q.stop();
    };
  }, [queue]);

  return <QueueContext.Provider value={queue}>{children}</QueueContext.Provider>;
}

/** The reader's event queue (null until signed in). */
export function useEventQueue(): EventQueue | null {
  return useContext(QueueContext);
}
