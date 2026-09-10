import { useEffect } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent, MutableRefObject } from 'react';
import type { Status } from '../types/chat';
import { AlertIcon, ArrowUpIcon, RotateIcon, XIcon } from '../lib/icons';

export interface ChatInputProps {
  value: string;
  onChange(v: string): void;
  onSend(text: string): void;
  onStop(): void;
  status: Status;
  error: string | null;
  onRetry(): void;
  onDismissError(): void;
  inputRef: MutableRefObject<HTMLTextAreaElement | null>;
}

export function ChatInput({ value, onChange, onSend, onStop, status, error, onRetry, onDismissError, inputRef }: ChatInputProps) {
  const busy = status === 'thinking' || status === 'streaming';
  const failed = status === 'error';
  const canSubmit = !busy && !failed && value.trim().length > 0;

  // Auto-resize up to 200px, then scroll.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [value, inputRef]);

  const submit = () => { if (canSubmit) onSend(value.trim()); };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer-outer">
      {failed && error && (
        <div className="errb" role="alert">
          <AlertIcon size={15} style={{ color: '#fbbf24' }} />
          <div className="msg-t">
            <span className="em">Connection failed. </span>
            <span className="ed">{error}</span>
          </div>
          <button type="button" onClick={onRetry} className="retry">
            <RotateIcon size={12} /><span>Retry</span>
          </button>
          <button type="button" onClick={onDismissError} className="dismiss" aria-label="Dismiss error">
            <XIcon size={13} />
          </button>
        </div>
      )}

      <div className={'composer' + (failed ? ' err' : '')}>
        <textarea
          ref={inputRef}
          rows={1}
          value={value}
          disabled={failed}
          autoComplete="off"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={failed ? 'Recover the connection to continue' : 'Message the model…'}
        />
        <div className="composer-foot">
          <div className="hint">Enter to send · Shift+Enter for a new line</div>
          <button type="button"
            onClick={busy ? onStop : submit}
            disabled={!busy && !canSubmit}
            aria-label={busy ? 'Stop generating' : 'Send message'}
            className={'send' + (busy || canSubmit ? ' on' : '')}>
            {busy ? <span className="stop-sq" /> : <ArrowUpIcon size={15} sw={2.5} />}
          </button>
        </div>
      </div>
      <p className="footnote">Parameters apply on the next request</p>
    </div>
  );
}