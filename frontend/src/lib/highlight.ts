/* Regex syntax highlighter — per-language configs with comments, strings,
   triple-quotes, numbers/hex, keywords, builtins, calls, and types. */

interface Tok { text: string; cls: string | null }
interface LangConf {
  line: string[];
  block?: [string, string];
  quotes: string;
  triple?: boolean;
  kw: Set<string>;
  builtins?: Set<string>;
  capType?: boolean;
  fnCall?: boolean;
  propColon?: boolean;
  tagLt?: boolean;
  hex?: boolean;
  ci?: boolean;
}

const KW = (s: string) => new Set(s.split(/\s+/).filter(Boolean));

const JS_KW = KW(`as async await break case catch class const continue debugger default delete do else enum export extends false finally for function get if implements import in instanceof interface let new null of package private protected public readonly return satisfies set static super switch this throw true try typeof undefined var void while with yield NaN Infinity`);
const JS_BI = KW(`console Object Array String Number Boolean Promise Map Set WeakMap WeakSet Symbol Date RegExp Error JSON Math window document localStorage fetch setTimeout clearTimeout setInterval clearInterval requestAnimationFrame process globalThis module exports require`);
const TS_EXTRA = KW(`abstract any bigint boolean declare keyof namespace never override string number symbol type unknown infer is asserts`);

const LANGS: Record<string, LangConf> = {
  javascript: { line: ['//'], block: ['/*', '*/'], quotes: `'"` + '`', kw: JS_KW, builtins: JS_BI, capType: true, fnCall: true },
  typescript: { line: ['//'], block: ['/*', '*/'], quotes: `'"` + '`', kw: new Set([...JS_KW, ...TS_EXTRA]), builtins: JS_BI, capType: true, fnCall: true },
  python: {
    line: ['#'], quotes: `'"`, triple: true, fnCall: true,
    kw: KW(`and as assert async await break class continue def del elif else except finally for from global if import in is lambda match case nonlocal not or pass raise return try while with yield None True False`),
    builtins: KW(`print len range list dict set tuple int str float bool complex bytes frozenset type object super isinstance issubclass getattr setattr hasattr callable enumerate zip map filter sorted reversed sum min max abs round pow divmod hex oct bin ord chr repr format vars dir id hash iter next open input self cls Exception ValueError TypeError KeyError IndexError RuntimeError StopIteration torch np numpy plt`),
  },
  bash: {
    line: ['#'], quotes: `'"`, fnCall: false,
    kw: KW(`if then else elif fi for while until do done case esac in function return local export readonly declare shift source alias exit trap set unset`),
    builtins: KW(`echo printf read cd pwd ls cp mv rm mkdir rmdir touch cat less head tail grep find sort uniq awk sed xargs chmod chown curl wget tar zip unzip ssh sudo su apt apt-get yum brew npm npx pnpm yarn pip pip3 python python3 node git docker make gcc`),
  },
  json: { line: [], quotes: '"', kw: KW('true false null') },
  css: { line: [], block: ['/*', '*/'], quotes: `'"`, propColon: true, hex: true, kw: KW('important') },
  yaml: { line: ['#'], quotes: `'"`, propColon: true, kw: new Set<string>() },
  html: { line: [], block: ['<!--', '-->'], quotes: `'"`, tagLt: true, kw: new Set<string>() },
  sql: {
    line: ['--'], block: ['/*', '*/'], quotes: `'"`, ci: true,
    kw: KW(`select from where and or not null is in like between as on join inner left right full outer cross union all distinct insert into values update set delete create table view index drop alter add column primary key foreign references constraint default check unique order by group having limit offset case when then else end exists asc desc`),
  },
  rust: {
    line: ['//'], block: ['/*', '*/'], quotes: `'"`, capType: true, fnCall: true,
    kw: KW(`as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self static struct super trait true type unsafe use where while`),
    builtins: KW(`println print vec String Vec Option Some None Result Ok Err Box Rc Arc HashMap HashSet format panic assert derive`),
  },
  go: {
    line: ['//'], block: ['/*', '*/'], quotes: `'"`, capType: true, fnCall: true,
    kw: KW(`break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var nil true false iota`),
    builtins: KW(`fmt println printf append len cap new make copy delete panic recover error string int int8 int16 int32 int64 float32 float64 bool byte rune`),
  },
  java: {
    line: ['//'], block: ['/*', '*/'], quotes: `'"`, capType: true, fnCall: true,
    kw: KW(`abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for goto if implements import instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try void volatile while true false null`),
    builtins: KW(`String Integer Long Double Float Boolean Character Object System Math List ArrayList Map HashMap Set HashSet Optional StringBuilder Thread`),
  },
  c: {
    line: ['//'], block: ['/*', '*/'], quotes: `'"`, capType: true, fnCall: true,
    kw: KW(`auto bool break case catch char class const constexpr continue decltype default delete do double else enum explicit extern false float for friend goto if inline int long mutable namespace new noexcept nullptr operator private protected public register return short signed sizeof static struct switch template this throw true try typedef typeid typename union unsigned using virtual void volatile while`),
    builtins: KW(`printf scanf malloc free memcpy memset strlen std cout cin endl vector string map set pair size_t uint32_t int64_t NULL`),
  },
};
LANGS.cpp = LANGS.c;

const LANG_ALIASES: Record<string, string> = {
  js: 'javascript', jsx: 'javascript', mjs: 'javascript', cjs: 'javascript',
  ts: 'typescript', tsx: 'typescript',
  py: 'python', sh: 'bash', shell: 'bash', zsh: 'bash',
  yml: 'yaml', scss: 'css', less: 'css',
  xml: 'html', svg: 'html',
  rs: 'rust', golang: 'go', 'c++': 'cpp', cxx: 'cpp', h: 'cpp', hpp: 'cpp', cs: 'java', kt: 'java',
  patch: 'diff',
};

function normalizeLang(raw?: string): string {
  if (!raw) return '';
  const l = raw.toLowerCase();
  return LANG_ALIASES[l] ?? l;
}

function tokenize(src: string, conf: LangConf): Tok[] {
  const out: Tok[] = [];
  const n = src.length;
  let i = 0;
  let prev = '';
  const push = (text: string, cls: string | null) => {
    if (text === '') return;
    out.push({ text, cls });
    if (text.trim() !== '') prev = text;
  };
  while (i < n) {
    const rest = src.slice(i);
    const ch = src[i];

    if (conf.block && rest.startsWith(conf.block[0])) {
      let end = src.indexOf(conf.block[1], i + conf.block[0].length);
      if (end === -1) end = n; else end += conf.block[1].length;
      push(src.slice(i, end), 'tk-c'); i = end; continue;
    }
    let comment = false;
    for (const lc of conf.line) {
      if (rest.startsWith(lc)) {
        const nl = src.indexOf('\n', i);
        const seg = nl === -1 ? rest : src.slice(i, nl);
        push(seg, 'tk-c'); i += seg.length; comment = true; break;
      }
    }
    if (comment) continue;

    if (conf.triple && (rest.startsWith('"""') || rest.startsWith("'''"))) {
      const q = rest.slice(0, 3);
      let end = src.indexOf(q, i + 3);
      if (end === -1) end = n; else end += 3;
      push(src.slice(i, end), 'tk-s'); i = end; continue;
    }

    if (conf.quotes.indexOf(ch) !== -1) {
      let j = i + 1;
      let end = n;
      while (j < n) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src[j] === ch) { end = j + 1; break; }
        if (src[j] === '\n' && ch !== '`') { end = j; break; }
        j++;
      }
      push(src.slice(i, Math.min(end, n)), 'tk-s'); i = Math.min(end, n); continue;
    }

    if (conf.hex && ch === '#') {
      const m = /^#[0-9a-fA-F]{3,8}/.exec(rest);
      if (m) { push(m[0], 'tk-n'); i += m[0].length; continue; }
    }

    if (ch >= '0' && ch <= '9') {
      const m = /^(?:0[xXbBoO][0-9a-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)[a-zA-Z_%]*/.exec(rest);
      if (m) { push(m[0], 'tk-n'); i += m[0].length; continue; }
    }

    const idm = /^[A-Za-z_$][A-Za-z0-9_$]*/.exec(rest);
    if (idm) {
      const w = idm[0];
      const lw = conf.ci ? w.toLowerCase() : w;
      const after = rest.slice(w.length);
      let cls: string | null = null;
      if (conf.kw.has(lw)) cls = 'tk-k';
      else if (conf.builtins && conf.builtins.has(lw)) cls = 'tk-b';
      else if (conf.tagLt && /<\/?$/.test(prev)) cls = 'tk-t';
      else if (conf.capType && /^[A-Z]/.test(w)) cls = 'tk-t';
      else if (conf.propColon && /^\s*:/.test(after)) cls = 'tk-a';
      else if (conf.fnCall && /^\s*\(/.test(after)) cls = 'tk-f';
      push(w, cls); i += w.length; continue;
    }

    const pm = /^[^\w\s]+/.exec(rest);
    if (pm) { push(pm[0], null); i += pm[0].length; continue; }

    const ws = /^\s+/.exec(rest);
    if (ws) { push(ws[0], null); i += ws[0].length; continue; }

    push(ch, null); i++;
  }
  return out;
}

export function highlightLines(code: string, language?: string): Tok[][] {
  const src = code.replace(/\n$/, '');
  if (language === 'diff' || language === 'patch') {
    return src.split('\n').map((l) => [{
      text: l,
      cls: l.startsWith('+') ? 'tk-add' : l.startsWith('-') ? 'tk-del' : l.startsWith('@@') ? 'tk-k' : null,
    }]);
  }
  const conf = language ? LANGS[normalizeLang(language)] : undefined;
  if (!conf) return src.split('\n').map((l) => [{ text: l, cls: null }]);
  const toks = tokenize(src, conf);
  const lines: Tok[][] = [[]];
  for (const t of toks) {
    if (t.text.indexOf('\n') === -1) { lines[lines.length - 1].push(t); continue; }
    const parts = t.text.split('\n');
    for (let p = 0; p < parts.length; p++) {
      if (p > 0) lines.push([]);
      if (parts[p] !== '') lines[lines.length - 1].push({ text: parts[p], cls: t.cls });
    }
  }
  return lines;
}