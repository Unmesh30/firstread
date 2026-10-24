// "Why am I seeing this?" for one article: /why?id=<uuid> (a query string, not a path
// segment: static hosting has no fallback for dynamic routes). It opens as a sheet over
// the feed; a deep link has the feed behind it too (the root layout's anchor).
import Head from 'expo-router/head';
import { router, useLocalSearchParams } from 'expo-router';

import { WHY_TITLE, WhySheet } from '@/features/why/WhySheet';
import { pageTitle } from '@/ui/Screen';

export default function Why() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  const close = () => (router.canGoBack() ? router.back() : router.replace('/feed'));
  return (
    <>
      <Head>
        <title>{pageTitle(WHY_TITLE)}</title>
      </Head>
      <WhySheet id={id} onClose={close} level={1} />
    </>
  );
}
