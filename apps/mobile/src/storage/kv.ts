// A small async key-value store: localStorage on web, AsyncStorage on native.
// Nothing here runs at import time (static rendering pre-renders routes in Node), and
// every web access is guarded: storage can be missing or throw (private windows).
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

const PREFIX = 'fr:';

export interface KV {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  remove(key: string): Promise<void>;
}

function webStorage(): Storage | null {
  try {
    return typeof window !== 'undefined' ? window.localStorage : null;
  } catch {
    return null;
  }
}

const web: KV = {
  async get(key) {
    try {
      return webStorage()?.getItem(PREFIX + key) ?? null;
    } catch {
      return null;
    }
  },
  async set(key, value) {
    try {
      webStorage()?.setItem(PREFIX + key, value);
    } catch {
      // Quota or a blocked store: the app keeps working without persistence.
    }
  },
  async remove(key) {
    try {
      webStorage()?.removeItem(PREFIX + key);
    } catch {
      // ignored, as above
    }
  },
};

const native: KV = {
  get: (key) => AsyncStorage.getItem(PREFIX + key),
  set: (key, value) => AsyncStorage.setItem(PREFIX + key, value),
  remove: (key) => AsyncStorage.removeItem(PREFIX + key),
};

export const kv: KV = Platform.OS === 'web' ? web : native;

/** An in-memory store for tests. */
export function memoryKV(initial: Record<string, string> = {}): KV & { data: Map<string, string> } {
  const data = new Map(Object.entries(initial));
  return {
    data,
    async get(key) {
      return data.get(key) ?? null;
    },
    async set(key, value) {
      data.set(key, value);
    },
    async remove(key) {
      data.delete(key);
    },
  };
}
