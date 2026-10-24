import { configErrors, isLocalUrl, type AppEnv } from '@/lib/env';

const base: AppEnv = {
  apiUrl: 'http://localhost:8000',
  authMode: 'dev',
  supabaseUrl: '',
  supabaseKey: '',
  siteUrl: 'https://x',
};

describe('configuration', () => {
  it('allows dev sign-in only against a local API', () => {
    expect(isLocalUrl('http://localhost:8010')).toBe(true);
    expect(isLocalUrl('http://127.0.0.1:8000')).toBe(true);
    expect(isLocalUrl('https://api.example.com')).toBe(false);
    expect(isLocalUrl('not a url')).toBe(false);
    expect(configErrors(base)).toEqual([]);
    expect(configErrors({ ...base, apiUrl: 'https://firstread-api.run.app' })).toHaveLength(1);
  });
  it('needs the Supabase URL and publishable key in supabase mode', () => {
    const sb: AppEnv = { ...base, apiUrl: 'https://api.example.com', authMode: 'supabase' };
    expect(configErrors(sb)).toHaveLength(1);
    expect(
      configErrors({
        ...sb,
        supabaseUrl: 'https://p.supabase.co',
        supabaseKey: 'sb_publishable_x',
      }),
    ).toEqual([]);
  });
});
