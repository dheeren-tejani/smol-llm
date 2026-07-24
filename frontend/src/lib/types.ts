export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface ChatParameters {
  max_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  repetition_penalty: number;
  range_epsilon: number;
}

export const DEFAULT_PARAMETERS: ChatParameters = {
  max_tokens: 256,
  temperature: 0.1,
  top_p: 0.9,
  top_k: 50,
  repetition_penalty: 1.3,
  range_epsilon: 0.1,
};

// Used ONLY for the unauthenticated warm-up ping to /health (see Index.tsx).
// Real chat traffic goes through /.netlify/functions/chat instead — that
// function holds the Modal URL + auth secret server-side, never exposed
// to the browser. Do NOT use this for /generate or /generate/stream calls.
export const MODAL_BASE_URL = import.meta.env.VITE_MODAL_BASE_URL || '';