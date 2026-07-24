import { useState, useCallback, useRef } from 'react';
import { ChatMessage, ChatParameters, DEFAULT_PARAMETERS, MODAL_BASE_URL } from '@/lib/types';

const WAKEUP_TIMEOUT_MS = 60_000;   // max total time we'll wait for the backend to wake up
const WAKEUP_POLL_INTERVAL_MS = 1500;

export function useChatState() {
  const [messages, setMessages]     = useState<ChatMessage[]>([]);
  const [parameters, setParameters] = useState<ChatParameters>(DEFAULT_PARAMETERS);
  const [isLoading, setIsLoading]   = useState(false);
  const [wakingUp, setWakingUp]     = useState(false);   // NEW — distinct from isLoading
  const [inputValue, setInputValue] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  // Polls /health until the model is loaded, or gives up after a time budget.
  // Returns true if the backend became ready, false if we timed out.
  const ensureBackendAwake = useCallback(async (signal: AbortSignal): Promise<boolean> => {
    const start = Date.now();

    while (Date.now() - start < WAKEUP_TIMEOUT_MS) {
      if (signal.aborted) return false;

      try {
        const res = await fetch(`${MODAL_BASE_URL}/health`, {
          cache: 'no-store',
          signal,
        });
        if (res.ok) {
          const body = await res.json();
          if (body.model_loaded) return true;
        }
        // 503 ("loading") or any non-200 — fall through and retry
      } catch {
        // Network error / cold container not yet accepting connections — retry
      }

      await new Promise(r => setTimeout(r, WAKEUP_POLL_INTERVAL_MS));
    }

    return false; // timed out
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id:        crypto.randomUUID(),
      role:      'user',
      content:   content.trim(),
      timestamp: new Date(),
    };

    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = {
      id:        assistantId,
      role:      'assistant',
      content:   '',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setInputValue('');
    setIsLoading(true);

    abortRef.current = new AbortController();

    try {
      // ── Step 1: make sure the backend is actually awake first ──────
      setWakingUp(true);
      const ready = await ensureBackendAwake(abortRef.current.signal);
      setWakingUp(false);

      if (!ready) {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: '⚠️ The model is taking longer than usual to start. Please try again in a moment.' }
              : m
          )
        );
        return;
      }

      // ── Step 2: now send the real request, backend should be warm ──
      const res = await fetch(`/.netlify/functions/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          prompt: content.trim(),
          ...parameters,
        }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) {
        // Rare race: became ready during our check, but slipped back to
        // loading/busy by the time this request landed. One quiet retry
        // rather than surfacing an error for a timing fluke.
        if (res.status === 503) {
          const retryRes = await fetch(`/.netlify/functions/chat`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ prompt: content.trim(), ...parameters }),
            signal: abortRef.current.signal,
          });
          if (!retryRes.ok) throw new Error(`Server error: ${retryRes.status}`);
          await streamResponse(retryRes, assistantId, setMessages);
          return;
        }
        throw new Error(`Server error: ${res.status}`);
      }

      await streamResponse(res, assistantId, setMessages);

    } catch (err) {
      if ((err as Error).name === 'AbortError') return;

      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? { ...m, content: '⚠️ Something went wrong. Please check your backend connection.' }
            : m
        )
      );
    } finally {
      setIsLoading(false);
      setWakingUp(false);
    }
  }, [isLoading, parameters, ensureBackendAwake]);

  const clearChat = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setInputValue('');
    setIsLoading(false);
    setWakingUp(false);
  }, []);

  return {
    messages,
    parameters,
    setParameters,
    isLoading,
    wakingUp,       // NEW — expose so the UI can show a distinct message
    inputValue,
    setInputValue,
    sendMessage,
    clearChat,
  };
}

// Extracted the SSE-reading loop into a helper so it's not duplicated
// between the normal path and the one-time 503 retry above.
async function streamResponse(
  res: Response,
  assistantId: string,
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
) {
  const reader  = res.body!.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) continue;

      const jsonStr = line.slice('data:'.length).trim();
      if (!jsonStr) continue;

      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(jsonStr);
      } catch {
        continue;
      }

      if (typeof msg.token === 'string') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId ? { ...m, content: m.content + (msg.token as string) } : m
          )
        );
      } else if (msg.done) {
        // done
      } else if (typeof msg.error === 'string') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId ? { ...m, content: `⚠️ ${msg.error}` } : m
          )
        );
      }
    }
  }
}