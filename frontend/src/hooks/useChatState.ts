import { useState, useCallback, useRef } from 'react';
import { ChatMessage, ChatParameters, DEFAULT_PARAMETERS, API_BASE_URL } from '@/lib/types';

export function useChatState() {
  const [messages, setMessages]     = useState<ChatMessage[]>([]);
  const [parameters, setParameters] = useState<ChatParameters>(DEFAULT_PARAMETERS);
  const [isLoading, setIsLoading]   = useState(false);
  const [inputValue, setInputValue] = useState('');
  const abortRef = useRef<AbortController | null>(null);

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

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      abortRef.current = new AbortController();

      const res = await fetch(`${API_BASE_URL}/generate/stream`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          prompt: content.trim(),
          ...parameters,
        }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      // Read the SSE stream
      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      let   buffer  = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // Append new chunk to buffer and process complete SSE lines
        buffer += decoder.decode(value, { stream: true });

        // SSE lines are separated by \n\n
        const parts = buffer.split('\n\n');
        // Keep the last (possibly incomplete) chunk in the buffer
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          // Each part looks like:  "data: {...}"
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
            // Append the new token to the assistant message
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content + msg.token }
                  : m
              )
            );
          } else if (msg.done) {
            // Generation complete — nothing extra needed, message is already built
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
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;

      // Replace the empty assistant placeholder with an error message
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
  }, [isLoading, parameters]);

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
    clearChat,
  };
}