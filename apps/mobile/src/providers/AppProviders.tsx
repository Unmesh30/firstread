import { onlineManager, QueryClientProvider } from '@tanstack/react-query';
import { useEffect, useState, useSyncExternalStore, type PropsWithChildren } from 'react';

import { createQueryClient } from '@/api/queries';
import { ThemeProvider } from '@/ui/theme';

import { wireOnlineManager } from './online';
import { SessionProvider } from './session';

export function AppProviders({ children }: PropsWithChildren) {
  const [client] = useState(createQueryClient);
  useEffect(wireOnlineManager, []);
  return (
    <ThemeProvider>
      <QueryClientProvider client={client}>
        <SessionProvider>{children}</SessionProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export function useOnline(): boolean {
  return useSyncExternalStore(
    (l) => onlineManager.subscribe(l),
    () => onlineManager.isOnline(),
    () => true,
  );
}
