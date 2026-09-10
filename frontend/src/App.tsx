import { useEffect, useRef, useState } from 'react';
import type { FC } from 'react';
import { useChatStream } from './hooks/useChatStream';
import { useBackendStatus } from './hooks/useBackendStatus';
import { useMediaQuery } from './hooks/useMediaQuery';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import {
  ArrowDownIcon, BracesIcon, FileIcon, LogoMark, NetworkIcon, PanelLeftIcon, BookOpenIcon, ListIcon, TableIcon, TrashIcon,
} from './lib/icons';
import type { IconProps } from './lib/icons';

const IS_MAC = typeof navigator !== 'undefined'
  && /mac|iphone|ipad|ipod/i.test(navigator.platform || navigator.userAgent);
const KBD_HINT = IS_MAC ? '⌘ K' : 'Ctrl K';

interface Starter { icon: FC<IconProps>; label: string; prompt: string }
const STARTERS: Starter[] = [
  { icon: NetworkIcon, label: 'Define Photosynthesis', prompt: 'Define Photosynthesis' },
  { icon: BracesIcon, label: 'Adding 2 numbers in Python', prompt: 'How to add 2 numbers in Python using + operator' },
  { icon: BookOpenIcon, label: 'Narrate a long story', prompt: 'Narrate a long story' },
  { icon: ListIcon, label: 'List 3 colors', prompt: 'List 3 colors' },
  { icon: TableIcon, label: "Physics vs Biology", prompt: "Difference between Physics and Biology"},
];

function EmptyState({ onStarter }: { onStarter(prompt: string): void }) {
  return (
    <div className="empty">
      <LogoMark size={56} glow />
      <h2>How can I help you today?</h2>
      <p>A 124M param LLaMA style LLM trained from scratch on random weights on 22B Cosmopedia tokens</p>
      <div className="starters">
        {STARTERS.map((s, i) => {
          const Ico = s.icon;
          return (
            <button type="button" key={s.label} className="starter"
              style={{ animationDelay: (0.08 + i * 0.06) + 's' }}
              onClick={() => onStarter(s.prompt)}>
              <Ico size={13.5} sw={1.75} />
              <span>{s.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function App() {
  const chat = useChatStream();
  const { status, newChat, stop, send, messages } = chat;
  const backend = useBackendStatus();

  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 768);
  const isMobile = useMediaQuery('(max-width: 767px)');
  const [draft, setDraft] = useState('');
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  /* scroll bookkeeping */
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const stickRef = useRef(true);
  const prevLenRef = useRef(messages.length);
  const [atBottom, setAtBottom] = useState(true);
  const [unread, setUnread] = useState(0);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    stickRef.current = bottom;
    setAtBottom(bottom);
    if (bottom) setUnread(0);
  };

  // Follow the stream only while pinned to the bottom.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const grew = messages.length > prevLenRef.current;
    prevLenRef.current = messages.length;
    if (grew && !stickRef.current) setUnread((u) => u + 1);
  }, [messages.length]);

  // Reset view when switching threads.
  useEffect(() => {
    prevLenRef.current = chat.messages.length;
    stickRef.current = true;
    setAtBottom(true);
    setUnread(0);
    const el = scrollRef.current;
    if (el) requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
  }, [chat.activeThreadId]); // eslint-disable-line react-hooks/exhaustive-deps

  const jumpToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickRef.current = true;
    setAtBottom(true);
    setUnread(0);
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  };

  const focusComposer = () => {
    if (window.matchMedia('(pointer: fine)').matches) inputRef.current?.focus();
  };

  const handleSend = (text: string) => {
      stickRef.current = true;
      setAtBottom(true);
      if (send(text)) setDraft(''); // clear the bar only if the send was accepted
  };

  const handleClearChat = () => {
      chat.clearChat();
      setDraft('');
  };

  const handleNewChat = () => {
    newChat();
    if (isMobile) setSidebarOpen(false);
    focusComposer();
  };

  const handleSelectThread = (id: string) => {
    chat.switchThread(id);
    if (isMobile) setSidebarOpen(false);
  };

  /* keyboard shortcuts: ⌘K / Ctrl+K new chat · Esc stop / close */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        handleNewChat();
      } else if (e.key === 'Escape') {
        if (status === 'thinking' || status === 'streaming') stop();
        else if (isMobile && sidebarOpen) setSidebarOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, isMobile, sidebarOpen, newChat, stop]);

  useEffect(() => { focusComposer(); }, []);

  const isEmpty = messages.length === 0;
  const lastIdx = messages.length - 1;
  const busy = status === 'thinking' || status === 'streaming';

  /* Header reflects the backend itself (mount ping + wake polling);
     request-level failures surface in the composer banner instead. */
  const waking = chat.waking || backend === 'waking';
  const dotOn = backend === 'ready' && !waking;
  const statusText = waking
    ? 'Waking up…'
    : backend === 'ready' ? 'Online'
    : backend === 'down' ? 'Offline'
    : 'Connecting…';

  return (
    <div data-app className="app">
      {isMobile && (
        <div className={'backdrop' + (sidebarOpen ? ' show' : '')} onClick={() => setSidebarOpen(false)} />
      )}
      <Sidebar
        open={sidebarOpen}
        isMobile={isMobile}
        onClose={() => setSidebarOpen(false)}
        params={chat.params}
        onParamChange={chat.setParam}
        onParamsReset={chat.resetParams}
        threads={chat.threads}
        activeThreadId={chat.activeThreadId}
        onNewChat={handleNewChat}
        onSelectThread={handleSelectThread}
        onDeleteThread={chat.deleteThread}
        newChatHint={KBD_HINT}
      />

      <main className="main">
        <header className="topbar">
          {(!sidebarOpen || isMobile) && (
            <button type="button" className="tb-btn" onClick={() => setSidebarOpen((o) => !o)} aria-label="Toggle sidebar">
              <PanelLeftIcon size={15} />
            </button>
          )}
          <div className="tb-id">
            <span className={'dot ' + (dotOn ? 'on' : 'off')} />
            <h1 className="tb-title">SLM Inference Engine</h1>
            <span className="tb-sep" aria-hidden="true">·</span>
            <span className={'tb-status' + (dotOn ? '' : ' off')}>{statusText}</span>
          </div>
          <div style={{ marginLeft: 'auto' }}>
            <button type="button" className="tb-clear" onClick={handleClearChat} disabled={isEmpty}>
              <TrashIcon size={13} />
              <span>Clear</span>
            </button>
          </div>
        </header>

        <div className="scrollwrap">
          <div ref={scrollRef} onScroll={handleScroll} role="log" aria-label="Conversation" className="scroll">
            {isEmpty ? (
              /* Starter pills send immediately — one tap fires the request. */
              <EmptyState onStarter={handleSend} />
            ) : (
              <div className="msg-col">
                {messages.map((m, i) => (
                  <ChatMessage
                    key={m.id}
                    message={m}
                    showTyping={status === 'thinking' && i === lastIdx}
                    showWaking={chat.waking && status === 'thinking' && i === lastIdx}
                    showCursor={status === 'streaming' && i === lastIdx && m.role === 'assistant'}
                    canRegenerate={status === 'idle' && i === lastIdx && m.role === 'assistant' && !!m.content}
                    onRegenerate={chat.regenerate}
                  />
                ))}
              </div>
            )}
          </div>

          {!atBottom && !isEmpty && (
            <button type="button" className="scrollpill" onClick={jumpToBottom}>
              {busy && <span className="streamdot" />}
              <span>Scroll to bottom</span>
              {unread > 0 && <span className="badge">{unread}</span>}
              <ArrowDownIcon size={13} />
            </button>
          )}
        </div>

        <ChatInput
          value={draft}
          onChange={setDraft}
          onSend={handleSend}
          onStop={stop}
          status={status}
          error={chat.error}
          onRetry={chat.retry}
          onDismissError={chat.dismissError}
          inputRef={inputRef}
        />
      </main>
    </div>
  );
}
