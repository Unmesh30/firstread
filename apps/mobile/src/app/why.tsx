// "Why am I seeing this?" for one article: /why?id=<uuid> (a query string, not a path
// segment: static hosting has no fallback for dynamic routes). Filled in by PR 4b.
import { Link, router, useLocalSearchParams } from 'expo-router';

import { Placeholder } from '@/features/Placeholder';
import { Button } from '@/ui';

export default function Why() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  return (
    <>
      <Placeholder
        title="Why am I seeing this?"
        body={id ? `Article ${id}` : 'No article given.'}
        testID="screen-why"
      />
      {router.canGoBack() ? (
        <Button variant="secondary" label="Close" onPress={() => router.back()} />
      ) : (
        <Link href="/feed" asChild>
          <Button variant="secondary" label="Back to your feed" />
        </Link>
      )}
    </>
  );
}
