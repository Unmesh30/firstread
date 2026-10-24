// Ports and URLs shared by the E2E config, runner and specs.
export const API_PORT = 8010;
export const WEB_PORT = 19006; // already in the API's CORS allowlist (Expo's web port)
export const API_URL = `http://localhost:${API_PORT}`;
export const WEB_URL = `http://localhost:${WEB_PORT}`;
export const BASE = '/firstread';
export const E2E_DB = 'firstread_e2e';
export const E2E_REDIS_DB = 3;
