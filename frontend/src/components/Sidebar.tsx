import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { GenerationParams, Thread } from '../types/chat';
import { PARAM_RANGES } from '../types/chat';
import { timeAgo } from '../lib/utils';
import { InfoIcon, PanelLeftIcon, PlusIcon, RotateIcon, TrashIcon } from '../lib/icons';

interface ParamDef {
  key: keyof GenerationParams;
  label: string;
  tip: string;
  format(v: number): string;
  min: number;
  max: number;
  step: number;
}

const PARAM_DEFS: ParamDef[] = [
  {
    ...PARAM_RANGES.max_output_tokens,
    key: 'max_output_tokens', label: 'Max Output Tokens',
    format: (v) => String(Math.round(v)),
    tip: 'Hard cap on response length. One token is roughly three-quarters of a word — lower it to make the model terse, raise it for long-form answers.',
  },
  {
    ...PARAM_RANGES.temperature,
    key: 'temperature', label: 'Temperature',
    format: (v) => v.toFixed(2),
    tip: 'Randomness of sampling. Low values stay deterministic and on-distribution; high values trade coherence for creativity.',
  },
  {
    ...PARAM_RANGES.top_p,
    key: 'top_p', label: 'Top P',
    format: (v) => v.toFixed(2),
    tip: 'Nucleus sampling — keeps the smallest set of tokens whose probabilities sum to P and samples only from it, cutting the unlikely tail.',
  },
  {
    ...PARAM_RANGES.top_k,
    key: 'top_k', label: 'Top K',
    format: (v) => String(Math.round(v)),
    tip: 'A hard cutoff: only the K most probable tokens are considered at each step. Pairs well with nucleus sampling.',
  },
  {
    ...PARAM_RANGES.repetition_penalty,
    key: 'repetition_penalty', label: 'Repetition Penalty',
    format: (v) => v.toFixed(2),
    tip: 'Penalizes tokens that already occurred in the context. 1.00 disables it; higher values discourage loops and parroting.',
  },
];

function InfoTip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const btnRef = useRef<HTMLButtonElement | null>(null);
  const show = () => {
    const r = btnRef.current?.getBoundingClientRect();
    if (r) setPos({ x: Math.max(8, Math.min(r.left - 30, window.innerWidth - 244)), y: r.bottom + 8 });
    setOpen(true);
  };
  return (
    <>
      <button ref={btnRef} type="button" className="info-btn" aria-label="More information"
        onMouseEnter={show} onMouseLeave={() => setOpen(false)}
        onFocus={show} onBlur={() => setOpen(false)}>
        <InfoIcon size={12} sw={2.25} />
      </button>
      {open && pos && createPortal(
        <div className="tip" role="tooltip" style={{ left: pos.x, top: pos.y }}>{text}</div>,
        document.body,
      )}
    </>
  );
}

function ParamControl({ def, value, onChange }: {
  def: ParamDef;
  value: number;
  onChange(key: keyof GenerationParams, v: number): void;
}) {
  const [draft, setDraft] = useState<string | null>(null);

  const clampStep = (n: number): number => {
    const stepped = Math.round(n / def.step) * def.step;
    return parseFloat(Math.min(def.max, Math.max(def.min, stepped)).toFixed(4));
  };
  const commit = (raw: string) => {
    const n = parseFloat(raw);
    if (Number.isFinite(n)) onChange(def.key, clampStep(n));
  };

  const pct = ((value - def.min) / (def.max - def.min)) * 100;

  return (
    <div className="param">
      <div className="prow">
        <div className="lbl">
          <span>{def.label}</span>
          <InfoTip text={def.tip} />
        </div>
        <input
          inputMode="decimal"
          className="num"
          value={draft ?? def.format(value)}
          onChange={(e) => { setDraft(e.target.value); commit(e.target.value); }}
          onFocus={(e) => { setDraft(def.format(value)); e.target.select(); }}
          onBlur={() => { if (draft !== null) commit(draft); setDraft(null); }}
          onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }}
        />
      </div>
      <input
        type="range" className="rng"
        min={def.min} max={def.max} step={def.step} value={value}
        aria-label={def.label}
        onChange={(e) => onChange(def.key, clampStep(parseFloat(e.target.value)))}
        style={{ '--p': pct + '%' } as React.CSSProperties}
      />
    </div>
  );
}

export interface SidebarProps {
  open: boolean;
  isMobile: boolean;
  params: GenerationParams;
  onParamChange(key: keyof GenerationParams, value: number): void;
  onParamsReset(): void;
  threads: Thread[];
  activeThreadId: string;
  onNewChat(): void;
  onSelectThread(id: string): void;
  onDeleteThread(id: string): void;
  onClose(): void;
  newChatHint: string;
}

function SidebarBody(p: SidebarProps) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => setTick((x) => x + 1), 30_000); // refresh relative times
    return () => window.clearInterval(t);
  }, []);

  return (
    <>
      <div className="side-top">
        <button type="button" className="newchat" onClick={p.onNewChat}>
          <PlusIcon size={15} sw={2.25} />
          <span>New chat</span>
          <span className="kbd">{p.newChatHint}</span>
        </button>
        <button type="button" className="collapse" onClick={p.onClose} aria-label="Collapse sidebar">
          <PanelLeftIcon size={15} />
        </button>
      </div>

      <div className="side-scroll">
        <div className="sect">
          <span>PARAMETERS</span>
          <button type="button" className="reset" onClick={p.onParamsReset}>
            <RotateIcon size={11} /><span>Reset</span>
          </button>
        </div>
        <div className="params">
          {PARAM_DEFS.map((d) => (
            <ParamControl key={d.key} def={d} value={p.params[d.key]} onChange={p.onParamChange} />
          ))}
        </div>

        <div className="sect"><span>HISTORY</span></div>
        <div className="threads">
          {p.threads.length === 0 ? (
            <p className="thread-empty">No conversations yet — threads are saved locally as you chat.</p>
          ) : p.threads.map((t) => {
            const active = t.id === p.activeThreadId;
            return (
              <div key={t.id} className={'thread' + (active ? ' active' : '')}>
                <button type="button" className="tt" onClick={() => p.onSelectThread(t.id)}>
                  <span>{t.title || 'Untitled chat'}</span>
                </button>
                <span className="ago">{timeAgo(t.createdAt)}</span>
                <button type="button" className="thdel" onClick={() => p.onDeleteThread(t.id)} aria-label="Delete chat">
                  <TrashIcon size={12.5} />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <div className="side-foot">SLM-1.4B · local inference · v2</div>
    </>
  );
}

export function Sidebar(p: SidebarProps) {
  const cls = 'sidebar' + (p.open ? ' open' : ' closed');
  return (
    <aside className={cls}>
      <div className="side-inner"><SidebarBody {...p} /></div>
    </aside>
  );
}