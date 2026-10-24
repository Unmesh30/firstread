// What the reader did in this session that the loaded pages don't know yet: liked
// stories, and hidden stories and sources (removed from view at once, while the undo
// toast is up and after). The server learns about them through the event queue.
import { create } from 'zustand';

interface FeedLocalState {
  liked: ReadonlySet<string>;
  hiddenArticles: ReadonlySet<string>;
  hiddenSources: ReadonlySet<string>;
  like(id: string): void;
  hide(scope: 'article' | 'source', key: string): void;
  unhide(scope: 'article' | 'source', key: string): void;
  reset(): void;
}

const add = (s: ReadonlySet<string>, k: string) => new Set(s).add(k);
const without = (s: ReadonlySet<string>, k: string) => {
  const n = new Set(s);
  n.delete(k);
  return n;
};
const field = (scope: 'article' | 'source') =>
  scope === 'article' ? ('hiddenArticles' as const) : ('hiddenSources' as const);

export const useFeedLocal = create<FeedLocalState>((set) => ({
  liked: new Set(),
  hiddenArticles: new Set(),
  hiddenSources: new Set(),
  like: (id) => set((s) => ({ liked: add(s.liked, id) })),
  hide: (scope, key) => set((s) => ({ [field(scope)]: add(s[field(scope)], key) })),
  unhide: (scope, key) => set((s) => ({ [field(scope)]: without(s[field(scope)], key) })),
  reset: () => set({ liked: new Set(), hiddenArticles: new Set(), hiddenSources: new Set() }),
}));
