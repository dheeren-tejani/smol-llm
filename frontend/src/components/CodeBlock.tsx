import { memo, useCallback, useMemo, useState } from 'react';
import { highlightLines } from '../lib/highlight';
import { copyText } from '../lib/utils';
import { CheckIcon, CopyIcon } from '../lib/icons';

interface CodeBlockProps {
  code: string;
  language?: string;
}

export const CodeBlock = memo(function CodeBlock({ code, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const lines = useMemo(() => highlightLines(code, language), [code, language]);

  const onCopy = useCallback(async () => {
    await copyText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }, [code]);

  return (
    <div className="cb">
      <div className="cb-head">
        <span className="cb-lang">{language || 'text'}</span>
        <button type="button" onClick={onCopy} className={'cb-copy' + (copied ? ' on' : '')}>
          <span className="pop" key={copied ? 'ok' : 'no'}>
            {copied ? <CheckIcon size={12} sw={2.5} /> : <CopyIcon size={12} />}
          </span>
          <span>{copied ? 'Copied' : 'Copy code'}</span>
        </button>
      </div>
      <div className="cb-body">
        <div className="cb-gutter" aria-hidden="true">
          {lines.map((_, idx) => <div key={idx}>{idx + 1}</div>)}
        </div>
        <div className="cb-scroll">
          <pre>
            {lines.map((toks, idx) => (
              <span className="cl" key={idx}>
                {toks.map((t, j) => t.cls ? <span key={j} className={t.cls}>{t.text}</span> : t.text)}
              </span>
            ))}
          </pre>
        </div>
      </div>
    </div>
  );
});