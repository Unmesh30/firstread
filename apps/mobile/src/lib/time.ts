// "12 minutes ago": how old a story is, in words a screen reader reads well too.

const units: [limitS: number, sizeS: number, name: string][] = [
  [3600, 60, 'minute'],
  [86_400, 3600, 'hour'],
  [Infinity, 86_400, 'day'],
];

export function timeAgo(iso: string, now: number = Date.now()): string {
  const at = Date.parse(iso);
  if (Number.isNaN(at)) return '';
  const s = Math.max(0, (now - at) / 1000);
  if (s < 60) return 'just now';
  for (const [limit, size, name] of units) {
    if (s < limit) {
      const n = Math.floor(s / size);
      return `${n} ${name}${n === 1 ? '' : 's'} ago`;
    }
  }
  return '';
}

/** Hours as words: "40 minutes", "5 hours", "2 days". */
export function ageInWords(hours: number): string {
  if (hours < 1) {
    const m = Math.max(1, Math.round(hours * 60));
    return `${m} minute${m === 1 ? '' : 's'}`;
  }
  if (hours < 48) {
    const h = Math.round(hours);
    return `${h} hour${h === 1 ? '' : 's'}`;
  }
  const d = Math.round(hours / 24);
  return `${d} days`;
}
