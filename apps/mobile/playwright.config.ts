// Web E2E against the local stack (Postgres + Redis from Docker Compose or CI service
// containers). Playwright starts the API on the seeded E2E database and the static
// server for dist/; `npm run e2e` seeds and builds first (e2e/run.mjs).
import { defineConfig, devices } from '@playwright/test';

import { API_PORT, API_URL, E2E_DB, E2E_REDIS_DB, WEB_PORT, WEB_URL } from './e2e/env';

const pg = new URL(
  process.env.DATABASE_URL ?? 'postgresql://firstread:firstread@localhost:5432/firstread',
);
pg.pathname = `/${E2E_DB}`;
const redis = new URL(process.env.REDIS_URL ?? 'redis://localhost:6379/0');
redis.pathname = `/${E2E_REDIS_DB}`;

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: { baseURL: WEB_URL, trace: 'retain-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: `uv run --frozen uvicorn firstread_api.main:app --host 127.0.0.1 --port ${API_PORT} --workers 1`,
      cwd: '../..',
      url: `${API_URL}/readyz`,
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
      env: {
        APP_ENV: 'local',
        LOG_LEVEL: 'WARNING',
        AUTH_DEV_BYPASS: 'true',
        DATABASE_URL: pg.toString(),
        REDIS_URL: redis.toString(),
        API_CORS_ORIGINS: WEB_URL,
        // Deadlines for a shared CI runner, not Cloud Run's shape.
        API_FEED_TIMEOUT_S: '5',
        API_STATE_TIMEOUT_S: '1',
        API_ENCODE_TIMEOUT_S: '1',
        API_CANDIDATES_TIMEOUT_S: '1',
        API_HYDRATE_TIMEOUT_S: '1',
        API_PERSIST_TIMEOUT_S: '1',
      },
    },
    {
      command: `node scripts/serve-pages.mjs --port ${WEB_PORT}`,
      url: `${WEB_URL}/firstread/`,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
