/**
 * useCancellableSend — shared "pause while thinking" hook for every chat
 * pane (GemmaChat + the 4 cloud providers: NIM / Claude / Gemini / OpenAI).
 *
 * Pattern:
 *   const { sending, begin, abort, finish } = useCancellableSend();
 *
 *   async function send() {
 *     const signal = begin();
 *     try {
 *       const reply = await providerChat(..., { signal });   // fetch passes signal
 *       // append assistant bubble
 *     } catch (e) {
 *       if (wasAborted(e)) return;                            // user hit Stop
 *       // show error banner
 *     } finally {
 *       finish();                                             // sending = false
 *     }
 *   }
 *
 *   // Send / Stop button toggle:
 *   <button onClick={sending ? abort : send}>{sending ? "Stop" : "Send"}</button>
 *
 * Behaviour:
 *   - ``begin()`` creates a fresh AbortController, aborts any prior one, and
 *     flips ``sending=true``. Returns the new signal, ready to pass to fetch.
 *   - ``abort()`` cancels the in-flight fetch. The caller's catch block
 *     receives an AbortError / DOMException; ``wasAborted()`` filters that
 *     case so it doesn't render as a user-facing error banner.
 *   - ``finish()`` clears ``sending`` regardless of outcome. Safe to call from
 *     the caller's ``finally`` even after abort (idempotent).
 *   - If the caller starts a new ``begin()`` while one is still running, the
 *     old controller is aborted first so we never have two parallel fetches
 *     from the same pane (matches user intent: "I changed my mind, ask this
 *     instead").
 */
import { useCallback, useEffect, useRef, useState } from "react";

export function useCancellableSend() {
  const controllerRef = useRef<AbortController | null>(null);
  const [sending, setSending] = useState(false);

  const begin = useCallback((): AbortSignal => {
    // Cancel any prior in-flight request — matches "I want to ask something
    // else NOW" intent even if the user didn't explicitly click Stop.
    controllerRef.current?.abort();
    const c = new AbortController();
    controllerRef.current = c;
    setSending(true);
    return c.signal;
  }, []);

  const abort = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setSending(false);
  }, []);

  const finish = useCallback(() => {
    controllerRef.current = null;
    setSending(false);
  }, []);

  // If the component unmounts mid-flight, drop the pending fetch so it
  // doesn't try to call setState on an unmounted tree.
  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
      controllerRef.current = null;
    };
  }, []);

  return { sending, begin, abort, finish };
}

/** Return true when an error object represents a user-initiated abort. */
export function wasAborted(e: unknown): boolean {
  if (e instanceof DOMException && e.name === "AbortError") return true;
  if (e instanceof Error && e.name === "AbortError") return true;
  // Some fetch polyfills stringify the message instead of preserving name.
  if (e instanceof Error && /aborted|cancel/i.test(e.message)) return true;
  return false;
}
