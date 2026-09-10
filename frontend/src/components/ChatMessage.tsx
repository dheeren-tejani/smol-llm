import { memo, useMemo, useState } from 'react';
import type { ChatMessage } from '../types/chat';
import { appendCursor, parseBlocks } from '../lib/markdown';
import { copyText, formatClock } from '../lib/utils';
import { CheckIcon, CopyIcon, LogoMark, RotateIcon } from '../lib/icons';

const MarkdownBody = memo(function MarkdownBody({ content, streaming }: { content: string; streaming: boolean }) {
  const blocks = useMemo(() => parseBlocks(content), [content]);
  return <div className="md">{streaming ? appendCursor(blocks) : blocks}</div>;
});

function TypingDots() {
  return (
    <div className="tdots" role="status" aria-label="Model is thinking">
      <i /><i /><i />
    </div>
  );
}

/** Shown instead of the dots while the backend cold-boots or the wake
    retry loop polls health — progress, not an error. */
function WakeIndicator() {
  return (
    <div className="tdots wake" role="status" aria-label="Backend is waking up">
      <i /><i /><i />
      <span className="wake-label">Waking the model…</span>
    </div>
  );
}

function CopyAction({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    await copyText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };
  return (
    <button type="button" onClick={onCopy} className="act">
      <span className="pop" key={copied ? 'ok' : 'no'}>
        {copied ? <CheckIcon size={12.5} sw={2.5} /> : <CopyIcon size={12.5} />}
      </span>
      <span>{copied ? 'Copied' : 'Copy'}</span>
    </button>
  );
}

export interface ChatMessageProps {
  message: ChatMessage;
  showTyping: boolean;
  showWaking: boolean;
  showCursor: boolean;
  canRegenerate: boolean;
  onRegenerate(): void;
}

export const ChatMessage = memo(function ChatMessage({ message, showTyping, showWaking, showCursor, canRegenerate, onRegenerate }: ChatMessageProps) {
  if (message.role === 'user') {
    return (
      <div className="msg u">
        <div className="wrap">
          <div className="bubble">{message.content}</div>
          {/* Reserved-height action row — appears on hover, zero layout shift */}
          <div className="actions">
            <div className="reveal"><CopyAction text={message.content} /></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="msg">
      <div className="avatar"><LogoMark size={14} /></div>
      <div style={{ minWidth: 0, flex: 1 }}>
        <div className="meta">
          <span className="name">SLM</span>
          <span className="time">{formatClock(message.timestamp)}</span>
        </div>
        {showWaking
          ? <WakeIndicator />
          : showTyping
            ? <TypingDots />
            : <MarkdownBody content={message.content} streaming={showCursor} />}
        <div className="actions">
          <div className="reveal">
            <CopyAction text={message.content} />
            {canRegenerate && (
              <button type="button" onClick={onRegenerate} className="act">
                <RotateIcon size={12} />
                <span>Regenerate</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});