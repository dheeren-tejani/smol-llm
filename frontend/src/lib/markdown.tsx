/* Markdown engine — renders straight to React elements.
   Blocks: headings, fences, hr, blockquotes, GFM tables, nested &
   task lists, display math, paragraphs. Inline: emphasis, strong,
   strikethrough, code spans, links/autolinks/images, inline math,
   escapes, hard breaks. */

import { cloneElement, createElement, isValidElement, useMemo } from 'react';
import type { Key, ReactElement, ReactNode } from 'react';
import katex from 'katex';
import { CodeBlock } from '../components/CodeBlock';

const LIST_RE = /^(\s*)([-*+]|\d{1,9}[.)])([ \t]+)(.*)$/;

function indentOf(s: string): number { return /^\s*/.exec(s)![0].length; }

function Tex({ tex, display }: { tex: string; display: boolean }) {
  const html = useMemo<string | null>(() => {
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore' });
    } catch {
      return null;
    }
  }, [tex, display]);
  if (html == null) {
    return display
      ? <div className="math-fb">{tex}</div>
      : <code className="math-fb">{tex}</code>;
  }
  return display
    ? <div className="ktx-d" dangerouslySetInnerHTML={{ __html: html }} />
    : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function parseInline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let plain = '';
  let k = 0;
  let i = 0;
  const n = text.length;
  const flush = () => { if (plain !== '') { out.push(plain); plain = ''; } };
  while (i < n) {
    const rest = text.slice(i);
    const ch = text[i];

    if (ch === '\n') {
      if (plain.endsWith('  ')) { plain = plain.replace(/[ \t]+$/, ''); flush(); out.push(<br key={'br' + (k++)} />); }
      else plain += ' ';
      i++; continue;
    }

    if (ch === '\\' && i + 1 < n && /[\\`*_{}\[\]()#+\-.!>$~|]/.test(text[i + 1])) {
      plain += text[i + 1]; i += 2; continue;
    }

    if (ch === '`') {
      const m = /^(`+)([\s\S]*?)\1(?!`)/.exec(rest);
      if (m) {
        flush();
        let c = m[2];
        if (c.length > 1 && c.startsWith(' ') && c.endsWith(' ') && c.trim() !== '') c = c.slice(1, -1);
        out.push(<code key={k++} className="ic">{c.replace(/\n/g, ' ')}</code>);
        i += m[0].length; continue;
      }
    }

    if (ch === '$' && !(i > 0 && /[\w$]/.test(text[i - 1]))) {
      const m = /^\$([^$\n]+)\$/.exec(rest);
      if (m && !/^\s|\s$/.test(m[1]) && m[1].trim() !== '') {
        flush();
        out.push(<Tex key={k++} tex={m[1]} display={false} />);
        i += m[0].length; continue;
      }
    }

    if (ch === '<') {
      const auto = /^<(https?:\/\/[^>\s]+|mailto:[^>\s]+)>/.exec(rest);
      if (auto) {
        flush();
        out.push(<a key={k++} href={auto[1]} target="_blank" rel="noopener noreferrer">{auto[1].replace(/^mailto:/, '')}</a>);
        i += auto[0].length; continue;
      }
      const brk = /^<br\s*\/?>$/i.exec(rest);
      if (brk) { flush(); out.push(<br key={k++} />); i += brk[0].length; continue; }
      const tag = /^<\/?[a-zA-Z][^>]*>?/.exec(rest);
      if (tag) { plain += tag[0]; i += tag[0].length; continue; }
    }

    if (ch === '[') {
      const m = /^\[([^\]]*)\]\(\s*<?([^()\s>]*)>?(?:\s+"[^"]*")?\s*\)/.exec(rest);
      if (m && m[2]) {
        flush();
        out.push(<a key={k++} href={m[2]} target="_blank" rel="noopener noreferrer">{parseInline(m[1])}</a>);
        i += m[0].length; continue;
      }
    }

    if (ch === '!' && text[i + 1] === '[') {
      const m = /^!\[([^\]]*)\]\(\s*<?([^()\s>]*)>?(?:\s+"[^"]*")?\s*\)/.exec(rest);
      if (m && m[2]) {
        flush();
        out.push(<a key={k++} href={m[2]} target="_blank" rel="noopener noreferrer">{m[1] || m[2]}</a>);
        i += m[0].length; continue;
      }
    }

    if (ch === '*') {
      let m = /^\*\*([\s\S]+?)\*\*(?!\*)/.exec(rest);
      if (m) { flush(); out.push(<strong key={k++}>{parseInline(m[1])}</strong>); i += m[0].length; continue; }
      m = /^\*(?!\s)([^*\n]+?)\*(?!\*)/.exec(rest);
      if (m) { flush(); out.push(<em key={k++}>{parseInline(m[1])}</em>); i += m[0].length; continue; }
    }

    if (ch === '_' && !(i > 0 && /\w/.test(text[i - 1]))) {
      let m = /^__([\s\S]+?)__(?!_)/.exec(rest);
      if (m) { flush(); out.push(<strong key={k++}>{parseInline(m[1])}</strong>); i += m[0].length; continue; }
      m = /^_(?!\s)([^_\n]+?)_(?!_)/.exec(rest);
      if (m) { flush(); out.push(<em key={k++}>{parseInline(m[1])}</em>); i += m[0].length; continue; }
    }

    if (ch === '~' && text[i + 1] === '~') {
      const m = /^~~(?!\s)([\s\S]+?)~~(?!~)/.exec(rest);
      if (m) { flush(); out.push(<del key={k++}>{parseInline(m[1])}</del>); i += m[0].length; continue; }
    }

    plain += ch;
    i++;
  }
  flush();
  return out;
}

function isTableSep(line: string): boolean {
  if (!line.includes('-') || !line.includes('|')) return false;
  const t = line.trim().replace(/^\|/, '').replace(/\|$/, '');
  const cells = t.split('|');
  return cells.length >= 1 && cells.every((c) => /^:?-+:?$/.test(c.trim()));
}

function splitRow(line: string): string[] {
  let t = line.trim();
  if (t.startsWith('|')) t = t.slice(1);
  if (t.endsWith('|') && !t.endsWith('\\|')) t = t.slice(0, -1);
  const cells: string[] = [];
  let cur = '';
  for (let k = 0; k < t.length; k++) {
    if (t[k] === '\\' && t[k + 1] === '|') { cur += '|'; k++; continue; }
    if (t[k] === '|') { cells.push(cur); cur = ''; continue; }
    cur += t[k];
  }
  cells.push(cur);
  return cells.map((c) => c.trim());
}

function parseTableBlock(lines: string[], start: number, key: Key): { node: ReactNode; next: number } {
  const header = splitRow(lines[start]);
  const aligns = splitRow(lines[start + 1]).map((c) => {
    const l = c.startsWith(':'), r = c.endsWith(':');
    return l && r ? 'center' : r ? 'right' : l ? 'left' : null;
  });
  let i = start + 2;
  const rows: string[][] = [];
  while (i < lines.length) {
    const l = lines[i];
    if (!l.trim() || !l.includes('|')) break;
    if (/^ {0,3}(`{3,}|~{3,}|#{1,6}\s|>)/.test(l) || LIST_RE.test(l)) break;
    rows.push(splitRow(l));
    i++;
  }
  const node = (
    <div className="table-wrap" key={key}>
      <table>
        <thead>
          <tr>{header.map((c, j) => {
            const align = aligns[j];
            return createElement('th', { key: j, style: align ? { textAlign: align } : undefined }, parseInline(c));
          })}</tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, j) => {
                const align = aligns[j];
                return createElement('td', { key: j, style: align ? { textAlign: align } : undefined }, parseInline(c));
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
  return { node, next: i };
}

function contentHasBlockStart(contentLines: string[]): boolean {
  return contentLines.some((l, idx) => idx > 0 && (
    /^ {0,3}(`{3,}|~{3,})/.test(l) ||
    /^ {0,3}#{1,6}\s/.test(l) ||
    /^ {0,3}>/.test(l) ||
    /^\s*\$\$/.test(l) ||
    LIST_RE.test(l) ||
    (l.includes('|') && idx + 1 < contentLines.length && isTableSep(contentLines[idx + 1]))
  ));
}

function parseListBlock(lines: string[], start: number, key: Key): { node: ReactNode; next: number } {
  const first = LIST_RE.exec(lines[start])!;
  const base = first[1].length;
  const ordered = /\d/.test(first[2]);
  const startNum = ordered ? parseInt(first[2], 10) : 1;
  const n = lines.length;

  let i = start;
  const block: string[] = [];
  while (i < n) {
    const line = lines[i];
    if (!line.trim()) {
      let j = i;
      while (j < n && !lines[j].trim()) j++;
      if (j >= n) break;
      const ni = indentOf(lines[j]);
      const nm = LIST_RE.exec(lines[j]);
      if (!(ni >= base + 2 || (nm !== null && ni <= base + 1))) break;
      for (let b = i; b < j; b++) block.push('');
      i = j;
      continue;
    }
    if (indentOf(line) < base) break;
    if (block.length > 0 && (
      /^ {0,3}(`{3,}|~{3,})/.test(line) ||
      /^ {0,3}#{1,6}\s/.test(line) ||
      /^ {0,3}>/.test(line) ||
      /^ {0,3}(?:\*[ \t]*){3,}$/.test(line) ||
      /^ {0,3}(?:-[ \t]*){3,}$/.test(line)
    )) break;
    block.push(line);
    i++;
  }

  interface RawItem { indent: number; marker: string; lines: string[]; blankInside: boolean }
  const items: RawItem[] = [];
  let cur: RawItem | null = null;
  for (const line of block) {
    const m = LIST_RE.exec(line);
    if (m && m[1].length <= base + 1) {
      cur = { indent: m[1].length, marker: m[2], lines: [m[4]], blankInside: false };
      items.push(cur);
    } else if (cur) {
      if (!line.trim()) { cur.blankInside = true; cur.lines.push(''); continue; }
      const contentIndent = cur.indent + cur.marker.length + 1;
      cur.lines.push(line.slice(Math.min(contentIndent, indentOf(line))));
    }
  }

  const loose = items.some((it) => it.blankInside);
  const isTask = !ordered && items.some((it) => /^\[[ xX]\][ \t]/.test(it.lines[0]));

  const rendered = items.map((it, idx) => {
    let contentLines = it.lines.slice();
    let checked: boolean | null = null;
    if (isTask) {
      const tm = /^\[([ xX])\][ \t]+(.*)$/.exec(contentLines[0]);
      if (tm) { checked = tm[1] !== ' '; contentLines[0] = tm[2]; }
    }
    const content = contentLines.join('\n');
    const kids = (loose || contentHasBlockStart(contentLines)) ? parseBlocks(content) : parseInline(content);
    return checked !== null
      ? <li key={idx} className="task"><input type="checkbox" className="cb" checked={checked} readOnly />{kids}</li>
      : <li key={idx}>{kids}</li>;
  });

  const listProps: { key: Key; start?: number; className?: string } = { key };
  if (ordered && startNum !== 1) listProps.start = startNum;
  if (isTask) listProps.className = 'tasklist';
  return { node: createElement(ordered ? 'ol' : 'ul', listProps, rendered), next: i };
}

export function parseBlocks(src: string): ReactNode[] {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const out: ReactNode[] = [];
  let i = 0;
  const n = lines.length;
  while (i < n) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    const fm = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (fm) {
      const fenceCh = fm[1][0];
      const info = fm[2].trim();
      const lang = info ? info.split(/\s+/)[0].toLowerCase() : '';
      i++;
      const buf: string[] = [];
      const closeRe = new RegExp('^ {0,3}' + (fenceCh === '`' ? '`' : '~') + '{3,}[ \\t]*$');
      while (i < n && !closeRe.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      out.push(<CodeBlock key={out.length} code={buf.join('\n')} language={lang || undefined} />);
      continue;
    }

    if (line.includes('|') && i + 1 < n && isTableSep(lines[i + 1])) {
      const r = parseTableBlock(lines, i, out.length);
      out.push(r.node); i = r.next; continue;
    }

    const hm = /^ {0,3}(#{1,6})(?:[ \t]+(.*))?$/.exec(line);
    if (hm) {
      const level = Math.min(hm[1].length, 4);
      const text = (hm[2] || '').replace(/\s+#+\s*$/, '').trim();
      out.push(createElement('h' + level, { key: out.length }, parseInline(text)));
      i++; continue;
    }

    if (/^ {0,3}(?:\*[ \t]*){3,}$/.test(line) || /^ {0,3}(?:-[ \t]*){3,}$/.test(line) || /^ {0,3}(?:_[ \t]*){3,}$/.test(line)) {
      out.push(<hr key={out.length} />); i++; continue;
    }

    if (/^ {0,3}>/.test(line)) {
      const buf: string[] = [];
      while (i < n && /^ {0,3}>/.test(lines[i])) {
        buf.push(lines[i].replace(/^ {0,3}>[ \t]?/, ''));
        i++;
      }
      out.push(<blockquote key={out.length}>{parseBlocks(buf.join('\n'))}</blockquote>);
      continue;
    }

    if (/^\s*\$\$/.test(line)) {
      const one = /^\s*\$\$([\s\S]+?)\$\$\s*$/.exec(line);
      if (one) { out.push(<Tex key={out.length} tex={one[1]} display={true} />); i++; continue; }
      i++;
      const buf: string[] = [];
      while (i < n && !/^\s*\$\$\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      out.push(<Tex key={out.length} tex={buf.join('\n')} display={true} />);
      continue;
    }

    if (LIST_RE.test(line)) {
      const r = parseListBlock(lines, i, out.length);
      out.push(r.node); i = r.next; continue;
    }

    const buf = [line];
    i++;
    while (i < n) {
      const l = lines[i];
      if (!l.trim()) break;
      if (/^ {0,3}(`{3,}|~{3,})/.test(l)) break;
      if (/^ {0,3}#{1,6}\s/.test(l)) break;
      if (/^ {0,3}>/.test(l)) break;
      if (/^\s*\$\$/.test(l)) break;
      if (LIST_RE.test(l)) break;
      if (/^ {0,3}(?:\*[ \t]*){3,}$/.test(l) || /^ {0,3}(?:-[ \t]*){3,}$/.test(l)) break;
      if (l.includes('|') && i + 1 < n && isTableSep(lines[i + 1])) break;
      buf.push(l); i++;
    }
    out.push(<p key={out.length}>{parseInline(buf.join('\n'))}</p>);
  }
  return out;
}

/* Streaming cursor: appended INSIDE the deepest last host element, so the
   blinking caret sits at the true tip of the stream — even inside list
   items, table cells, or emphasized text. */
const CURSOR_EL = <span className="stream-cursor" key="cursor" />;

export function appendCursor(node: ReactNode): ReactNode {
  if (node == null || node === false || node === true) return node;
  if (Array.isArray(node)) {
    if (node.length === 0) return CURSOR_EL;
    const copy = node.slice();
    copy[copy.length - 1] = appendCursor(node[node.length - 1]);
    return copy;
  }
  if (typeof node === 'string' || typeof node === 'number') return [node, CURSOR_EL];
  if (isValidElement(node)) {
    const props = (node as ReactElement<{ children?: ReactNode; dangerouslySetInnerHTML?: unknown }>).props;
    if (typeof node.type === 'string' && props.dangerouslySetInnerHTML === undefined && props.children != null) {
      return cloneElement(node, {}, appendCursor(props.children));
    }
    return [node, CURSOR_EL];
  }
  return node;
}