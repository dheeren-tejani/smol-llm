import { useState, useRef, useCallback } from 'react';

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export function useChatState() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (prompt: string, systemPrompt?: string) => {
    if (!prompt.trim()) return;

    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const userMessage: Message = { role: 'user', content: prompt };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Point to the Netlify Proxy Function
      const response = await fetch('/.netlify/functions/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          system: systemPrompt,
          // Optional: Add max_tokens, temperature, etc. here if your UI supports it
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error('Rate limit exceeded. Please wait a moment and try again.');
        }
        if (response.status === 503) {
          throw new Error('Server is currently at capacity or starting up. Please retry shortly.');
        }
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Response body is not readable');

      const decoder = new TextDecoder('utf-8');
      let done = false;
      let buffer = '';

      // Add a placeholder message for the assistant's streaming text
      setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          
          // SSE events are separated by double newlines
          const parts = buffer.split('\n\n');
          // Keep the last partial chunk in the buffer
          buffer = parts.pop() || '';

          for (const part of parts) {
            if (part.startsWith('data: ')) {
              const dataStr = part.slice(6);
              try {
                const data = JSON.parse(dataStr);
                
                if (data.error) {
                  throw new Error(data.error);
                }

                if (data.token) {
                  setMessages((prev) => {
                    const newMessages = [...prev];
                    const lastIndex = newMessages.length - 1;
                    newMessages[lastIndex] = {
                      ...newMessages[lastIndex],
                      content: newMessages[lastIndex].content + data.token,
                    };
                    return newMessages;
                  });
                }
              } catch (e) {
                console.warn('Failed to parse SSE JSON chunk', e);
              }
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Request cancelled');
      } else {
        console.error('Chat error:', err);
        setError(err.message || 'An unexpected error occurred.');
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, []);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
    }
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    stopGeneration,
    clearChat,
  };
}