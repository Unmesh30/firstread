import { useQuery } from '@tanstack/react-query';

import { fetchers, keys, staleTimes } from '@/api/queries';
import type { MeResponse } from '@/api/schema-types';
import { useApi } from '@/providers/session';

export function useMe() {
  const api = useApi();
  return useQuery({
    queryKey: keys.me,
    queryFn: () => fetchers.me(api!),
    enabled: api != null,
    staleTime: staleTimes.me,
  });
}

/** A reader with no topics and no demo persona hasn't been onboarded yet. */
export function needsOnboarding(me: Pick<MeResponse, 'topics' | 'active_persona'>): boolean {
  return me.topics.length === 0 && !me.active_persona;
}

export function useTopics() {
  const api = useApi();
  return useQuery({
    queryKey: keys.topics,
    queryFn: () => fetchers.topics(api!),
    enabled: api != null,
    staleTime: staleTimes.topics,
  });
}
