/* Backend lifecycle indicator. Pings /api/health once on mount (waking a
   cold/sleeping backend), polls every few seconds until the model reports
   loaded, then STOPS — an idle backend should be allowed to fall asleep
   again. Re-probes when the tab becomes visible so a backend that slept
   while the user was away is re-warmed before they type. */

import { useEffect, useRef, useState } from 'react';
import { checkBackendHealth } from '../services/chatService';

export type BackendState = 'probing' | 'waking' | 'ready' | 'down';

const POLL_WHILE_WAKING_MS = 5000;
const GIVE_UP_AFTER_MS = 180_000; // never ready AND nothing ever succeeded

export function useBackendStatus(): BackendState {
  const [state, setState] = useState<BackendState>('probing');
  const stateRef = useRef(state);
  useEffect(() => { stateRef.current = state; }, [state]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let everReady = false;
    const startedAt = Date.now();

    const stop = () => {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
    };
    const schedule = (ms: number) => {
      stop();
      timer = setTimeout(() => { void run(); }, ms);
    };

    const run = async () => {
      const h = await checkBackendHealth();
      if (cancelled) return;
      if (h.modelLoaded) {
        everReady = true;
        setState('ready');
        stop(); // let the backend sleep again once idle — no keep-alive beacon
        return;
      }
      if (h.reachable) {
        // Container is up, model still loading.
        setState('waking');
        schedule(POLL_WHILE_WAKING_MS);
        return;
      }
      // Unreachable: most likely cold-booting (this very poll is what wakes
      // it). Only call it "down" if it has NEVER come up within the window.
      if (!everReady && Date.now() - startedAt > GIVE_UP_AFTER_MS) {
        setState('down');
        stop();
        return;
      }
      setState('waking');
      schedule(POLL_WHILE_WAKING_MS);
    };

    const onVisible = () => {
      if (document.visibilityState !== 'visible') return;
      // One re-probe on focus: if the backend slept, this wakes it; if it's
      // warm, the loop stops immediately after a single request.
      if (stateRef.current !== 'probing') void run();
    };

    document.addEventListener('visibilitychange', onVisible);
    void run();

    return () => {
      cancelled = true;
      stop();
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, []);

  return state;
}