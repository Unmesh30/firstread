// What a reader can do with a story: open it, like it, hide it (or its source), and
// each becomes an event in the queue.
import { useQueryClient } from '@tanstack/react-query';
import { Platform } from 'react-native';

import { keys } from '@/api/queries';
import { useEventQueue } from '@/events/EventsProvider';
import { openUrl } from '@/lib/platformKit';
import { hideButtonId } from '@/ui/ArticleCard';
import { useToasts } from '@/ui/toast';

import { useFeedLocal } from './store';
import type { FeedRow } from './useFeed';

export type HideScope = 'article' | 'source';

/** A press can reach a link twice (key-up and the browser's click): one click event. */
const OPEN_DEBOUNCE_MS = 1000;
const lastOpen = new Map<string, number>();

export function hideMessage(row: FeedRow, scope: HideScope): string {
  return scope === 'article' ? 'Story hidden' : `Stories from ${row.item.source_name} hidden`;
}

export function useArticleActions() {
  const queue = useEventQueue();
  const qc = useQueryClient();
  const local = useFeedLocal.getState;

  const base = (row: FeedRow) => ({
    article_id: row.item.id,
    position: row.position,
    request_id: row.requestId,
  });

  return {
    /** On web the link itself opens the tab; this records the click. */
    open(row: FeedRow) {
      const now = Date.now();
      if (now - (lastOpen.get(row.item.id) ?? 0) < OPEN_DEBOUNCE_MS) return;
      lastOpen.set(row.item.id, now);
      queue?.enqueue({ type: 'click', ...base(row) });
      if (Platform.OS !== 'web') openUrl(row.item.url);
    },

    /** There is no unlike in v1: a second press only explains that. */
    like(row: FeedRow) {
      if (local().liked.has(row.item.id)) {
        useToasts.getState().show({ message: 'Already liked. Likes can’t be undone yet' });
        return;
      }
      local().like(row.item.id);
      queue?.enqueue({ type: 'like', ...base(row) });
    },

    /**
     * Hidden from view now; the hide event is queued only when the undo toast closes
     * (or at once if the page goes to the background meanwhile).
     */
    hide(row: FeedRow, scope: HideScope): number {
      const key = scope === 'article' ? row.item.id : row.item.source_domain;
      // One undo at a time: a newer hide commits the one before it.
      useToasts.getState().commitPending();
      local().hide(scope, key);
      return useToasts.getState().show({
        message: hideMessage(row, scope),
        actionLabel: 'Undo',
        commitOnHide: true,
        onAction: () => local().unhide(scope, key),
        // A keyboard Undo puts focus back on the card the reader hid from.
        returnFocusTo: hideButtonId(row.item.id),
        onExpire: () => {
          queue?.enqueue({ type: 'hide', scope, ...base(row) });
          // Account lists hidden sources; it refetches /v1/me next time it's shown.
          if (scope === 'source')
            void qc.invalidateQueries({ queryKey: keys.me, refetchType: 'none' });
        },
      });
    },
  };
}
