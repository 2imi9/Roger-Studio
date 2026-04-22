/**
 * Safe sessionStorage wrapper with quota-management.
 *
 * The audit caught this: every heavy writer (5 chat components' message
 * arrays + App.tsx's labeledFeatures collection) was doing
 *   try { sessionStorage.setItem(k, v); } catch { /_ noop _/ }
 * which silently swallowed QuotaExceededError. Users with long-running
 * sessions hit the ~5 MB per-origin limit and their most recent messages
 * stopped persisting across reloads — no error, no warning, just a
 * confusing "where did my chat go?" report.
 *
 * safeSetItem attempts the write, and on quota failure:
 *   1. Logs a throttled warning (at most once per minute) so the console
 *      isn't spammed by the 100 setItem calls a single chat turn makes.
 *   2. Evicts the oldest chat-history key and retries once. "Oldest" is
 *      determined by a per-key ``__written_at`` timestamp map kept in a
 *      single metadata slot.
 *   3. Returns false if the retry also failed — caller decides whether to
 *      drop the write or degrade gracefully.
 *
 * We target chat-history keys specifically for eviction because they're
 * the only things large enough to reclaim meaningful bytes. Small writes
 * (sidebar flags, selected model names, api keys) aren't worth evicting
 * and aren't registered with the metadata map.
 */

const META_KEY = "roger.ssmeta.writtenAt.v1";
const WARN_THROTTLE_MS = 60_000;

let lastWarnedAt = 0;

type WrittenAtMap = Record<string, number>;

function readMeta(): WrittenAtMap {
  try {
    const raw = sessionStorage.getItem(META_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}

function writeMeta(meta: WrittenAtMap): void {
  try {
    sessionStorage.setItem(META_KEY, JSON.stringify(meta));
  } catch {
    // Metadata write itself hit quota — nothing more to do, the quota
    // path below will try to evict on the next real write.
  }
}

function isQuotaError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  // Browsers disagree on the error name and code. Firefox uses
  // NS_ERROR_DOM_QUOTA_REACHED (code 1014); Chromium uses QuotaExceededError
  // (code 22); Safari sometimes throws a plain DOMException with the name.
  const name = err.name;
  const anyErr = err as { code?: number };
  return (
    name === "QuotaExceededError" ||
    name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    anyErr.code === 22 ||
    anyErr.code === 1014
  );
}

function warnOnce(msg: string): void {
  const now = Date.now();
  if (now - lastWarnedAt < WARN_THROTTLE_MS) return;
  lastWarnedAt = now;
  console.warn(`[sessionStorage] ${msg}`);
}

function evictOldestTrackedKey(excludeKey: string): boolean {
  const meta = readMeta();
  const candidates = Object.entries(meta).filter(([k]) => k !== excludeKey);
  if (candidates.length === 0) return false;
  candidates.sort(([, a], [, b]) => a - b);
  const [oldestKey] = candidates[0];
  try {
    sessionStorage.removeItem(oldestKey);
    delete meta[oldestKey];
    writeMeta(meta);
    warnOnce(`quota hit — evicted ${oldestKey} to free space`);
    return true;
  } catch {
    return false;
  }
}

/**
 * Attempt to persist ``value`` at ``key`` in sessionStorage.
 *
 * Pass ``trackForEviction: true`` for large, expendable writes (chat
 * histories, labeled-feature collections) so the wrapper knows it can
 * evict this key when another large write hits the quota wall. Small
 * writes (flags, preferences) should omit it — they're cheap to keep
 * and annoying to lose.
 *
 * Returns ``true`` when the value is persisted, ``false`` when both the
 * initial write and the retry-after-eviction both failed (at which point
 * the caller should assume the data is in-memory only).
 */
export function safeSetItem(
  key: string,
  value: string,
  opts: { trackForEviction?: boolean } = {},
): boolean {
  try {
    sessionStorage.setItem(key, value);
    if (opts.trackForEviction) {
      const meta = readMeta();
      meta[key] = Date.now();
      writeMeta(meta);
    }
    return true;
  } catch (err) {
    if (!isQuotaError(err)) {
      // Non-quota errors: iOS Safari private mode throws SecurityError,
      // some embedded webviews disable storage entirely. Nothing we can
      // do, swallow quietly (matches prior behavior).
      return false;
    }
    if (!evictOldestTrackedKey(key)) {
      warnOnce(`quota hit for ${key} — no evictable keys, write dropped`);
      return false;
    }
    try {
      sessionStorage.setItem(key, value);
      if (opts.trackForEviction) {
        const meta = readMeta();
        meta[key] = Date.now();
        writeMeta(meta);
      }
      return true;
    } catch {
      warnOnce(`quota hit for ${key} — retry after eviction failed`);
      return false;
    }
  }
}
