import { useState, useCallback, useRef } from 'react';
import { ChatMessage, ChatParameters, DEFAULT_PARAMETERS, MODAL_BASE_URL } from '@/lib/types';

const WAKEUP_TIMEOUT_MS = 60_000;
const WAKEUP_POLL_INTERVAL_MS = 1500;

export function useChatState() {
  const [messages, setMessages]     = useState<ChatMessage[]>([]);
  const [parameters, setParameters] = useState<ChatParameters>(DEFAULT_PARAMETERS);
  const [isLoading, setIsLoading]   = useState(false);
  const [inputValue, setInputValue] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  // Polls /health until the model is loaded, or gives up after a time budget.
  // Runs BEFORE the real request so a cold/sleeping backend never causes a
  // failed generate call — the user just sees the normal loading state for
  // a bit longer instead of an error.
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
        // network error / cold container not yet accepting connections — retry
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

    // Placeholder assistant message that we'll fill in token by token
    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = {
      id:        assistantId,
      role:      'assistant',
      content:   '',
      timestamp: new Date(),
    };

    // Snapshot the history BEFORE this turn — used to build the full
    // conversation payload sent to the backend, since setMessages below
    // won't be reflected in this closure synchronously.
    const priorMessages = messages
      .filter(m => (m.role === 'user' || m.role === 'assistant') && m.content.trim() !== '')
      .map(m => ({ role: m.role, content: m.content }));

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setInputValue('');
    setIsLoading(true);

    abortRef.current = new AbortController();

    try {
      // Step 1: make sure the backend is actually awake before sending the
      // real request — folded into isLoading so the UI shows nothing new.
      const ready = await ensureBackendAwake(abortRef.current.signal);

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

      const payload = JSON.stringify({
        messages: [...priorMessages, { role: 'user', content: content.trim() }],
        ...parameters,
      });

      // Step 2: send the real request, backend should be warm by now.
      const res = await fetch(`/.netlify/functions/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    payload,
        signal: abortRef.current.signal,
      });

      if (!res.ok) {
        if (res.status === 503) {
          // Rare race: became ready during our check, slipped back to
          // loading/busy by the time this request landed. One quiet retry
          // rather than surfacing an error for a timing fluke.
          const retryRes = await fetch(`/.netlify/functions/chat`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    payload,
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
    }
  }, [isLoading, parameters, messages, ensureBackendAwake]);

  const stopGeneration = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
      setIsLoading(false);
    }
  }, []);

  const clearChat = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setInputValue('');
    setIsLoading(false);
  }, []);

  return {
    messages,
    parameters,
    setParameters,
    isLoading,
    inputValue,
    setInputValue,
    sendMessage,
    stopGeneration,
    clearChat,
  };
}

// Reads the SSE stream and appends tokens to the assistant placeholder as
// they arrive. Extracted so both the normal path and the one-time 503
// retry above share the exact same logic.
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

    // SSE events are separated by double newlines
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
            m.id === assistantId
              ? { ...m, content: m.content + (msg.token as string) }
              : m
          )
        );
      } else if (msg.done) {
        // Generation complete — nothing extra needed, message already built.
      } else if (typeof msg.error === 'string') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: `⚠️ ${msg.error}` }
              : m
          )
        );
      }
    }
  }
}