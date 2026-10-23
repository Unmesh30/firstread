// FirstRead feed load test (P3e). Local Docker stack only, never production.
//
// Constant arrival rate: 50 requests/s for 60 s, then 100 requests/s for 60 s.
// Mix: 60% first feed pages, 25% cursor pages, 15% event batches (10 impressions,
// one in five with a click). Readers and their ES256 tokens come from
// loadtest/seed_local.py; the API verifies them against AUTH_LOCAL_JWKS_FILE.
//
//   docker run --rm --network firstread_default -v "$PWD/loadtest:/lt" \
//     grafana/k6:2.3.0 run --out csv=/lt/results/raw.csv /lt/k6/feed.js
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE = __ENV.BASE || "http://firstread-api-lt:8000";
const users = new SharedArray("users", () => JSON.parse(open("/lt/.tokens/tokens.json")));

// Server time: the sum of the Server-Timing stages the feed reports.
const serverMs = new Trend("server_ms", true);
const errors = new Rate("errors");
const degraded = new Counter("degraded_responses");
const degradedBy = {};
for (const r of ["redis", "encoder", "ann", "pools", "timeout"]) {
  degradedBy[r] = new Counter(`degraded_${r}`);
}

const STATS = ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max", "count"];
export const options = {
  summaryTrendStats: STATS,
  scenarios: {
    rps50: {
      executor: "constant-arrival-rate",
      rate: 50,
      timeUnit: "1s",
      duration: "60s",
      preAllocatedVUs: 50,
      maxVUs: 400,
    },
    rps100: {
      executor: "constant-arrival-rate",
      rate: 100,
      timeUnit: "1s",
      duration: "60s",
      startTime: "60s",
      preAllocatedVUs: 100,
      maxVUs: 600,
    },
  },
  // Sub-metrics per scenario and request kind (reported, not used to fail the run).
  thresholds: {
    "http_req_duration{scenario:rps50}": ["p(95)<1000000"],
    "http_req_duration{scenario:rps100}": ["p(95)<1000000"],
    "server_ms{scenario:rps50}": ["p(95)<1000000"],
    "server_ms{scenario:rps100}": ["p(95)<1000000"],
    "http_req_duration{kind:page1}": ["p(95)<1000000"],
    "http_req_duration{kind:cursor}": ["p(95)<1000000"],
    "http_req_duration{kind:events}": ["p(95)<1000000"],
    "errors{scenario:rps50}": ["rate<1"],
    "errors{scenario:rps100}": ["rate<1"],
  },
};

const state = {}; // per VU: uid -> {cursor, items}

function uuid4() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function serverTime(res) {
  const h = res.headers["Server-Timing"];
  if (!h) return null;
  let ms = 0;
  for (const part of h.split(",")) {
    const m = part.match(/dur=([0-9.]+)/);
    if (m) ms += parseFloat(m[1]);
  }
  return ms;
}

function feed(u, cursor, kind) {
  const q = cursor ? `?limit=20&cursor=${cursor}` : "?limit=20";
  const res = http.get(`${BASE}/v1/feed${q}`, {
    headers: { Authorization: `Bearer ${u.token}` },
    tags: { kind },
  });
  const ok = check(res, { "feed 200": (r) => r.status === 200 });
  errors.add(!ok, { kind });
  const st = serverTime(res);
  if (st !== null) serverMs.add(st, { kind });
  if (!ok) return;
  const body = res.json();
  if (body.degraded.length) {
    degraded.add(1, { kind });
    for (const r of body.degraded) if (degradedBy[r]) degradedBy[r].add(1, { kind });
  }
  state[u.uid] = { cursor: body.next_cursor, items: body.items.map((i) => i.id) };
}

function events(u) {
  const s = state[u.uid];
  const items = s.items.slice(0, 10);
  const evs = items.map((id, k) => ({
    event_id: uuid4(),
    type: "impression",
    article_id: id,
    position: k,
  }));
  if (Math.random() < 0.2) {
    evs.push({ event_id: uuid4(), type: "click", article_id: pick(items) });
  }
  const res = http.post(`${BASE}/v1/events`, JSON.stringify({ events: evs }), {
    headers: { Authorization: `Bearer ${u.token}`, "Content-Type": "application/json" },
    tags: { kind: "events" },
  });
  const ok = check(res, { "events 200": (r) => r.status === 200 });
  errors.add(!ok, { kind: "events" });
}

export default function () {
  const r = Math.random();
  const known = Object.keys(state).filter((k) => state[k].items.length);
  if (r < 0.6 || !known.length) {
    feed(pick(users), null, "page1");
    return;
  }
  const uid = pick(known);
  const u = users.find((x) => x.uid === uid);
  if (r < 0.85 && state[uid].cursor) feed(u, state[uid].cursor, "cursor");
  else events(u);
}

export function handleSummary(data) {
  return { "/lt/results/k6-summary.json": JSON.stringify(data, null, 1) };
}
