/* Message state, SSE lifecycle, abort, parameters, and thread
   persistence (localStorage). Token appends are rAF-batched into a
   single setState per frame so the message tree re-renders at display
   cadence, not at network cadence.

   Cold-start resilience: a send that fails with a wake-retryable error
   (proxy 502/504, model loading, capacity) polls /api/health until the
   model is loaded, then transparently re-sends — the UI shows a
   "waking" indicator instead of an error, and Stop aborts at any phase. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ChatMessage, GenerationParams, Status, StreamPayload, Thread } from '../types/chat';
import { DEFAULT_PARAMS, sanitizeParams } from '../types/chat';
import {
  consumeStream, describeError, isWakeRetryable, waitForBackendReady,
} from '../services/chatService';
import { deriveTitle, mkMsg, uid } from '../lib/utils';

const LS_THREADS = 'slm-console:threads:v1';
const LS_PARAMS = 'slm-console:params:v1';
const EMPTY_MSGS: ChatMessage[] = [];

const MAX_SEND_ATTEMPTS = 3;      // initial send + up to 2 wake-retries
const SLOW_FIRST_TOKEN_MS = 5000; // dots → "Waking the model…" after this
const WAKE_POLL_MS = 3000;
const WAKE_MAX_WAIT_MS = 60_000;

function loadThreads(): Thread[] {
  try {
    const raw = localStorage.getItem(LS_THREADS);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((t): t is Thread =>
      !!t && typeof (t as Thread).id === 'string' && Array.isArray((t as Thread).messages));
  } catch { return []; }
}

function loadParams(): GenerationParams {
  try { return sanitizeParams(JSON.parse(localStorage.getItem(LS_PARAMS) ?? '{}')); }
  catch { return { ...DEFAULT_PARAMS }; }
}

function buildPayload(msgs: ChatMessage[]): Array<{ role: string; content: string }> {
  return msgs
    .filter((m) => (m.role === 'user' || m.role === 'assistant') && m.content.trim())
    .slice(-16) // small context window — keep the tail
    .map((m) => ({ role: m.role, content: m.content }));
}

export function useChatStream() {
  const [threads, setThreads] = useState<Thread[]>(loadThreads);
  const [activeId, setActiveId] = useState<string>(() => uid());
  const [params, setParams] = useState<GenerationParams>(loadParams);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [waking, setWaking] = useState(false);

  const threadsRef = useRef<Thread[]>(threads);
  const activeIdRef = useRef<string>(activeId);
  const paramsRef = useRef<GenerationParams>(params);
  const statusRef = useRef<Status>(status);
  useEffect(() => { threadsRef.current = threads; }, [threads]);
  useEffect(() => { activeIdRef.current = activeId; }, [activeId]);
  useEffect(() => { paramsRef.current = params; }, [params]);
  useEffect(() => { statusRef.current = status; }, [status]);

  const abortRef = useRef<AbortController | null>(null);
  const bufferRef = useRef('');
  const rafRef = useRef(0);
  const targetRef = useRef<{ threadId: string; stubId: string } | null>(null);
  const firstTokenRef = useRef(false);

  /* persistence */
  useEffect(() => {
    try { localStorage.setItem(LS_THREADS, JSON.stringify(threads)); } catch { /* quota */ }
  }, [threads]);
  useEffect(() => {
    try { localStorage.setItem(LS_PARAMS, JSON.stringify(params)); } catch { /* quota */ }
  }, [params]);
  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    abortRef.current?.abort();
  }, []);

  const messages = useMemo<ChatMessage[]>(
    () => threads.find((t) => t.id === activeId)?.messages ?? EMPTY_MSGS,
    [threads, activeId],
  );

  const flush = useCallback(() => {
    rafRef.current = 0;
    const chunk = bufferRef.current;
    bufferRef.current = '';
    const target = targetRef.current;
    if (!chunk || !target) return;
    const { threadId, stubId } = target;
    setThreads((prev) => prev.map((t) => t.id !== threadId ? t : {
      ...t,
      messages: t.messages.map((m) => m.id !== stubId ? m : { ...m, content: m.content + chunk }),
    }));
  }, []);

  const scheduleFlush = useCallback(() => {
    if (!rafRef.current) rafRef.current = requestAnimationFrame(flush);
  }, [flush]);

  const pruneStub = useCallback((threadId: string, stubId: string) => {
    setThreads((prev) => prev.map((t) => t.id !== threadId ? t : {
      ...t,
      messages: t.messages.filter((m) => !(m.id === stubId && !m.content.trim())),
    }));
  }, []);

  const runGeneration = useCallback(async (threadId: string, payload: StreamPayload, stubId: string) => {
    targetRef.current = { threadId, stubId };
    bufferRef.current = '';
    setError(null);

    let outcome: 'done' | 'aborted' | 'error' = 'error';
    let finalError: string | null = null;

    for (let attempt = 1; attempt <= MAX_SEND_ATTEMPTS; attempt++) {
      firstTokenRef.current = false;
      setStatus('thinking');
      const controller = new AbortController();
      abortRef.current = controller;

      // A cold boot makes the request hang with no first token — after
      // SLOW_FIRST_TOKEN_MS, flip the UI to "waking" so it reads as
      // progress rather than silence. Cleared on the first token.
      const wakeHint = window.setTimeout(() => {
        if (!firstTokenRef.current && !controller.signal.aborted) setWaking(true);
      }, SLOW_FIRST_TOKEN_MS);

      try {
        await consumeStream(payload, controller.signal, {
          onToken: (token) => {
            if (!token) return;
            if (!firstTokenRef.current) {
              firstTokenRef.current = true;
              setStatus('streaming');
              setWaking(false);
              window.clearTimeout(wakeHint);
            }
            bufferRef.current += token;
            scheduleFlush();
          },
        });
        window.clearTimeout(wakeHint);
        setWaking(false);
        flush();
        outcome = 'done';
        break;
      } catch (err) {
        window.clearTimeout(wakeHint);
        flush(); // keep whatever partial text arrived
        setWaking(false);

        if (controller.signal.aborted || (err as { name?: string } | null)?.name === 'AbortError') {
          outcome = 'aborted'; // user-initiated stop — not an error
          break;
        }

        // Cold-start / loading / capacity: poll health (free on the
        // backend) until the model is ready, then re-send. Only retry if
        // nothing streamed yet — mid-stream failures keep partial text
        // and surface as errors instead.
        if (isWakeRetryable(err) && !firstTokenRef.current && attempt < MAX_SEND_ATTEMPTS) {
          setWaking(true);
          const ready = await waitForBackendReady(controller.signal, {
            pollMs: WAKE_POLL_MS,
            maxWaitMs: WAKE_MAX_WAIT_MS,
          });
          setWaking(false);
          if (controller.signal.aborted) { outcome = 'aborted'; break; }
          if (ready) continue; // backend is warm — re-send
          finalError = 'The backend is taking longer than expected to wake up. Give it a minute and try again.';
          outcome = 'error';
          break;
        }

        finalError = describeError(err);
        outcome = 'error';
        break;
      }
    }

    if (outcome === 'done' || outcome === 'aborted') {
      setStatus('idle');
    } else {
      setStatus('error');
      setError(finalError ?? 'Unexpected error while streaming the response.');
    }

    pruneStub(threadId, stubId); // drop the stub if nothing ever streamed
    abortRef.current = null;
    targetRef.current = null;
    bufferRef.current = '';
    setWaking(false);
  }, [flush, scheduleFlush, pruneStub]);

  const send = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    if (statusRef.current === 'thinking' || statusRef.current === 'streaming') return;

    const threadId = activeIdRef.current;
    const existing = threadsRef.current.find((t) => t.id === threadId);
    const userMsg = mkMsg('user', trimmed);
    const payload: StreamPayload = {
      messages: buildPayload([...(existing?.messages ?? []), userMsg]),
      parameters: { ...paramsRef.current },
    };
    const stub = mkMsg('assistant', '');

    setError(null);
    setThreads((prev) => {
      const idx = prev.findIndex((t) => t.id === threadId);
      if (idx === -1) {
        return [{ id: threadId, title: deriveTitle(trimmed), createdAt: Date.now(), messages: [userMsg, stub] }, ...prev];
      }
      const updated = { ...prev[idx], messages: [...prev[idx].messages, userMsg, stub] };
      return [updated, ...prev.filter((t) => t.id !== threadId)]; // most-recent-first
    });
    void runGeneration(threadId, payload, stub.id);
  }, [runGeneration]);

  const stop = useCallback(() => { abortRef.current?.abort(); }, []);

  const regenerate = useCallback(() => {
    if (statusRef.current === 'thinking' || statusRef.current === 'streaming') return;
    const threadId = activeIdRef.current;
    const thread = threadsRef.current.find((t) => t.id === threadId);
    if (!thread) return;
    const msgs = [...thread.messages];
    while (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') msgs.pop();
    if (msgs.length === 0 || msgs[msgs.length - 1].role !== 'user') return;

    const payload: StreamPayload = { messages: buildPayload(msgs), parameters: { ...paramsRef.current } };
    const stub = mkMsg('assistant', '');
    setError(null);
    setThreads((prev) => prev.map((t) => t.id !== threadId ? t : { ...t, messages: [...msgs, stub] }));
    void runGeneration(threadId, payload, stub.id);
  }, [runGeneration]);

  const retry = useCallback(() => {
    if (statusRef.current !== 'error') return;
    setError(null);
    regenerate();
  }, [regenerate]);

  const dismissError = useCallback(() => {
    setError(null);
    if (statusRef.current === 'error') setStatus('idle');
  }, []);

  const clearChat = useCallback(() => {
    if (targetRef.current) abortRef.current?.abort();
    setError(null);
    setStatus('idle');
    setWaking(false);
    const threadId = activeIdRef.current;
    setThreads((prev) => prev.filter((t) => t.id !== threadId));
    setActiveId(uid());
  }, []);

  const newChat = useCallback(() => { setActiveId(uid()); setError(null); }, []);

  const switchThread = useCallback((id: string) => { setActiveId(id); setError(null); }, []);

  const deleteThread = useCallback((id: string) => {
    if (targetRef.current?.threadId === id) abortRef.current?.abort();
    setThreads((prev) => prev.filter((t) => t.id !== id));
    if (activeIdRef.current === id) {
      setActiveId(uid());
      setError(null);
      setStatus('idle');
    }
  }, []);

  const setParam = useCallback((key: keyof GenerationParams, value: number) => {
    setParams((p) => (p[key] === value ? p : { ...p, [key]: value }));
  }, []);

  const resetParams = useCallback(() => setParams({ ...DEFAULT_PARAMS }), []);

  return {
    threads, activeThreadId: activeId, messages, params, status, error, waking,
    setParam, resetParams, send, stop, retry, regenerate, dismissError,
    clearChat, newChat, switchThread, deleteThread,
  };
}