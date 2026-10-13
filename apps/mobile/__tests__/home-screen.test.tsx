import { render, screen } from '@testing-library/react-native';

import { HomeScreen } from '@/components/home-screen';

describe('HomeScreen', () => {
  it('shows the FirstRead title as a header', async () => {
    await render(<HomeScreen />);
    expect(screen.getByRole('header', { name: 'FirstRead' })).toBeTruthy();
  });

  it('shows the tagline', async () => {
    await render(<HomeScreen />);
    expect(screen.getByText(/before anyone has clicked/)).toBeTruthy();
  });
});
