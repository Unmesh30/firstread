// TanStack Query pauses requests while offline; expo-network tells it when that is
// (navigator.onLine and the online/offline events on web).
import { onlineManager } from '@tanstack/react-query';
import * as Network from 'expo-network';

let wired = false;

export function wireOnlineManager(): void {
  if (wired) return;
  wired = true;
  onlineManager.setEventListener((setOnline) => {
    const sub = Network.addNetworkStateListener((state) => setOnline(state.isConnected !== false));
    return () => sub.remove();
  });
}
