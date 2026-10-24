// After `expo export -p web`: make dist/ ready for GitHub Pages.
//   - 404.html: Pages serves it for unknown paths; Expo writes the not-found route as
//     +not-found.html.
//   - .nojekyll must be present (copied from public/), or Pages hides _expo/.
import { copyFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const dist = process.argv[2] ?? 'dist';
const notFound = join(dist, '+not-found.html');
if (!existsSync(notFound))
  throw new Error(`${notFound} is missing: is app/+not-found.tsx still there?`);
copyFileSync(notFound, join(dist, '404.html'));
if (!existsSync(join(dist, '.nojekyll')))
  throw new Error('dist/.nojekyll is missing (public/.nojekyll)');
console.log('postexport: 404.html written, .nojekyll present');
