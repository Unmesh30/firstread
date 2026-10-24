// A section that a later P4 PR fills in (4b: feed and Why; 4c: the rest).
import { Screen, Text } from '@/ui';

export function Placeholder({
  title,
  body,
  testID,
}: {
  title: string;
  body: string;
  testID?: string;
}) {
  return (
    <Screen title={title} testID={testID}>
      <Text variant="headline" serif role="heading" aria-level={1}>
        {title}
      </Text>
      <Text tone="muted">{body}</Text>
    </Screen>
  );
}
