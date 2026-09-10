/* Domain types, parameter ranges & defaults. Range epsilon removed.
   Slider bounds aligned with the backend's GenerateRequest validation
   (max_tokens ≤ 1024, temperature ≥ 0.01) so no slider position can
   ever produce a 422. */

export interface GenerationParams {
  max_output_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  repetition_penalty: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface StreamPayload {
  messages: Array<{ role: string; content: string }>;
  parameters: GenerationParams;
}

export interface Thread {
  id: string;
  title: string;
  createdAt: number;
  messages: ChatMessage[];
}

export type Status = 'idle' | 'thinking' | 'streaming' | 'error';

export const PARAM_RANGES = {
  max_output_tokens: { min: 16, max: 1024, step: 1 },  // backend caps at 1024
  temperature: { min: 0.01, max: 2, step: 0.01 },     // backend requires >= 0.01
  top_p: { min: 0, max: 1, step: 0.01 },
  top_k: { min: 1, max: 100, step: 1 },
  repetition_penalty: { min: 1, max: 2, step: 0.05 },
} as const;

export const DEFAULT_PARAMS: GenerationParams = {
  max_output_tokens: 256,
  temperature: 0.10,
  top_p: 0.90,
  top_k: 50,
  repetition_penalty: 1.30,
};

/** Clamp + step-align anything coming back from localStorage.
   Previously-stored out-of-bounds values (e.g. max tokens 2048 from the
   old ranges) are clamped to the new bounds on load. */
export function sanitizeParams(raw: unknown): GenerationParams {
  const out = { ...DEFAULT_PARAMS };
  if (!raw || typeof raw !== 'object') return out;
  const src = raw as Record<string, unknown>;
  (Object.keys(PARAM_RANGES) as Array<keyof GenerationParams>).forEach((key) => {
    const range = PARAM_RANGES[key];
    const n = Number(src[key]);
    if (!Number.isFinite(n)) return;
    const stepped = Math.round(n / range.step) * range.step;
    out[key] = parseFloat(Math.min(range.max, Math.max(range.min, stepped)).toFixed(4));
  });
  return out;
}