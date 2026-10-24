// npm run e2e [-- --skip-build] [playwright args]
//   1. seed the E2E database and Redis db (services/api/scripts/e2e_seed.py);
//   2. export the web app against the E2E API (EXPO_PUBLIC_API_URL) and run postexport;
//   3. run Playwright (which starts the API and the static server).
import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2);
const skipBuild = args.includes('--skip-build');
const rest = args.filter((a) => a !== '--skip-build');

function run(cmd, cmdArgs, opts = {}) {
  const r = spawnSync(cmd, cmdArgs, { stdio: 'inherit', ...opts });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

run('uv', ['run', '--frozen', 'python', 'services/api/scripts/e2e_seed.py'], { cwd: '../..' });
if (!skipBuild) {
  run('npx', ['expo', 'export', '-p', 'web'], {
    env: {
      ...process.env,
      EXPO_PUBLIC_API_URL: 'http://localhost:8010',
      EXPO_PUBLIC_AUTH_MODE: 'dev',
    },
  });
  run('node', ['scripts/postexport.mjs']);
}
run('npx', ['playwright', 'test', ...rest]);
