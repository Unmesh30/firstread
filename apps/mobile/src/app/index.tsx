// The entry gate: onboarding for a new reader, the feed otherwise. Rendered statically
// as a skeleton; the decision happens on the client once the session is ready.
import { Redirect } from 'expo-router';

import { needsOnboarding, useMe } from '@/features/me';
import { useSession } from '@/providers/session';
import { Banner, ErrorState, LoadingList, Screen } from '@/ui';

export default function Index() {
  const session = useSession();
  const me = useMe();
  if (session.status === 'error') {
    return (
      <Screen title="Loading">
        <Banner tone="danger" message={session.message} testID="config-error" />
      </Screen>
    );
  }
  if (me.data) return <Redirect href={needsOnboarding(me.data) ? '/onboarding' : '/feed'} />;
  if (me.error) {
    return (
      <Screen title="Loading">
        <ErrorState error={me.error} onRetry={() => void me.refetch()} testID="gate-error" />
      </Screen>
    );
  }
  return (
    <Screen title="Loading">
      <LoadingList label="Loading your feed" />
    </Screen>
  );
}
