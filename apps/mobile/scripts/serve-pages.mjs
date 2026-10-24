// Serve dist/ the way GitHub Pages serves this project site, for local preview and E2E:
//   - everything lives under /firstread (experiments.baseUrl in app.json);
//   - /firstread/x serves x.html, x/index.html or the file x itself;
//   - unknown paths get 404.html with status 404 (Pages has no SPA fallback);
//   - query strings are ignored for lookup (/firstread/why?id=… serves why.html).
// No dependencies. Usage: node scripts/serve-pages.mjs [--port 19006] [--dir dist]
import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const PORT = Number(opt('port', '19006'));
const ROOT = resolve(opt('dir', 'dist'));
const BASE = '/firstread';

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.ttf': 'font/ttf',
  '.txt': 'text/plain; charset=utf-8',
  '.map': 'application/json',
};

function isFile(p) {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
}

export function resolvePath(root, urlPath) {
  if (!urlPath.startsWith(`${BASE}/`) && urlPath !== BASE) return null;
  let rel = decodeURIComponent(urlPath.slice(BASE.length)) || '/';
  rel = normalize(rel).replace(/^(\.\.[/\\])+/, '');
  const base = join(root, rel);
  if (!base.startsWith(root)) return null;
  for (const candidate of [base, `${base}.html`, join(base, 'index.html')]) {
    if (isFile(candidate)) return candidate;
  }
  return null;
}

function send(res, status, file) {
  res.writeHead(status, {
    'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream',
    'Cache-Control': 'no-cache',
  });
  createReadStream(file).pipe(res);
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (url.pathname === '/' || url.pathname === BASE) {
    res.writeHead(301, { Location: `${BASE}/` });
    res.end();
    return;
  }
  const file = resolvePath(ROOT, url.pathname);
  if (file) send(res, 200, file);
  else if (isFile(join(ROOT, '404.html'))) send(res, 404, join(ROOT, '404.html'));
  else {
    res.writeHead(404).end('not found');
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`serving ${ROOT} at http://localhost:${PORT}${BASE}/`);
});
