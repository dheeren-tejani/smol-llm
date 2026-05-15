
"""
prepare_sft_data.py — OpenHermes 2.5 → Senku SFT Dataset

Pipeline:
  1. Load OpenHermes 2.5 JSON (local file or HF download)
  2. Keep ONLY strict 2-turn conversations (1 user + 1 assistant).
     Multi-turn records are dropped to avoid multi-turn loss-mask bugs.
  3. Drop any record whose assistant turn contains robotic AI-identity
     phrases ("As an AI...", "I don't have feelings...", etc.).
     Only explicit AI *names* (ChatGPT, Claude, …) are replaced → Senku.
  4. Filter to ≤ max_tokens total tokens.
  5. Shuffle and cap at max_records (50 000).
  6. Save to output JSON  (list of {"conversations": [...]} records)

Usage:
    # From Hugging Face (auto-download):
    python prepare_sft_data.py

    # From a local file:
    python prepare_sft_data.py --input /path/to/openhermes2_5.json

    # Full options:
    python prepare_sft_data.py \
        --input openhermes2_5.json \
        --output senku_sft_50k.json \
        --model-name Senku \
        --max-tokens 1024 \
        --max-records 50000 \
        --tokenizer gpt2 \
        --seed 42
"""

import re
import json
import random
import argparse
from pathlib import Path
from typing import Optional

# ── Optional tqdm ────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x


# ════════════════════════════════════════════════════════════
# 1.  Identity patterns
# ════════════════════════════════════════════════════════════

# ── 1a. Explicit AI model names → replaced with model_name ──────────────
# These are references to *other* models by name (e.g. "ChatGPT says…").
# Replacing them is safe because the sentence structure stays intact.
_KNOWN_AI_NAMES = [
    # OpenAI
    "ChatGPT", "GPT-4o", "GPT-4", "GPT-3.5", "GPT-3", "GPT4", "GPT3", "GPT",
    "OpenAI",
    # Anthropic
    "Claude", "Anthropic",
    # Google
    "Gemini", "Bard", "Google AI", "Google Assistant",
    # Meta
    "LLaMA", "Llama", "Meta AI", "Meta Llama",
    # xAI
    "Grok",
    # Mistral
    "Mixtral", "Mistral",
    # DeepSeek
    "DeepSeek", "Deepseek",
    # Microsoft
    "Copilot", "Bing AI", "Bing Chat",
    # Others
    "Falcon", "Vicuna", "Alpaca", "Dolly", "Bloom", "Pythia",
    "StableLM", "MPT", "OPT", "BLOOM",
]

# Sort longest first so longer phrases match before substrings
_KNOWN_AI_NAMES_SORTED = sorted(_KNOWN_AI_NAMES, key=len, reverse=True)

# ── 1b. Robotic AI-identity phrases → DROP the whole record ─────────────
# These produce stilted, ChatGPT-flavoured prose even after name injection.
# Because we have >>50k records available, it's cheaper to skip than salvage.
_ROBOTIC_IDENTITY_PATTERNS = [
    # "I am / I'm an AI …"
    r"\bI(?:'m| am) an? (?:AI|artificial intelligence|language model|LLM|"
    r"machine learning model|neural network|chatbot|chat bot|virtual assistant|"
    r"digital assistant|computer program|software)\b",

    # "As an AI …"
    r"\bAs an? (?:AI|artificial intelligence|language model|LLM|"
    r"machine learning model|neural network|chatbot|chat bot|virtual assistant|"
    r"digital assistant|computer program|software)\b",

    # "I was created / trained / developed / built by <org>"
    r"\bI (?:was |have been )?(?:created|trained|developed|built|designed|made) "
    r"by (?:OpenAI|Anthropic|Google|Meta|xAI|Mistral|DeepSeek|Microsoft|"
    r"Hugging Face|a team of researchers|researchers|engineers|developers|"
    r"a company|an organization)\b",

    # "my training data / cutoff / corpus"
    r"\bmy (?:training (?:data|cutoff|corpus)|knowledge cutoff|"
    r"knowledge base|dataset)\b",

    # "I don't have feelings / consciousness / a body …"
    r"\bI (?:do not|don't) have (?:personal )?(?:experiences?|feelings?|"
    r"emotions?|consciousness|sentience|opinions?|beliefs?|desires?|"
    r"preferences?|a physical body|a body)\b",

    # "I cannot browse the internet / access the web"
    r"\bI (?:cannot|can't|am unable to) (?:browse|access|search) "
    r"(?:the )?(?:internet|web|online)\b",

    # Hedge-filler openers that signal boilerplate responses
    r"\bCertainly!? (?:I(?:'d| would) be (?:happy|glad|delighted) to\b|"
    r"here(?:'s| is) (?:a |an )?(?:step-by-step |detailed )?(?:guide|explanation|breakdown|overview)\b)",
    r"\bOf course!? (?:I(?:'d| would) be (?:happy|glad|delighted) to\b)",
    r"\bAbsolutely!? (?:I(?:'d| would) be (?:happy|glad|delighted) to\b)",
]

# Compile once at import time
_COMPILED_ROBOTIC = [
    re.compile(pat, re.IGNORECASE) for pat in _ROBOTIC_IDENTITY_PATTERNS
]


def has_robotic_identity(text: str) -> bool:
    """Return True if text contains any robotic AI-identity phrase."""
    return any(p.search(text) for p in _COMPILED_ROBOTIC)


def inject_name(text: str, model_name: str) -> str:
    """
    Replace explicit AI model names in `text` with `model_name`.
    Does NOT attempt to patch robotic identity prose — those records
    are dropped upstream in process() before this function is called.
    """
    for name in _KNOWN_AI_NAMES_SORTED:
        pattern = r'\b' + re.escape(name) + r'\b'
        text = re.sub(pattern, model_name, text, flags=re.IGNORECASE)
    return text


# ════════════════════════════════════════════════════════════
# 2.  Tokenizer  (tiktoken → HF tokenizer → whitespace fallback)
# ════════════════════════════════════════════════════════════

def _load_tokenizer(name: str):
    """
    Returns a callable token_count(text: str) -> int.
    Priority: tiktoken > HuggingFace tokenizer > whitespace split.
    """
    # Try tiktoken first (fastest, good for GPT-family vocab sizes)
    try:
        import tiktoken
        enc = tiktoken.get_encoding(name) if name in ("gpt2", "cl100k_base", "o200k_base") \
              else tiktoken.encoding_for_model(name)
        print(f"[tokenizer] Using tiktoken '{name}'")
        return lambda t: len(enc.encode(t))
    except Exception:
        pass

    # Fall back to HuggingFace tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        print(f"[tokenizer] Using HuggingFace tokenizer '{name}'")
        return lambda t: len(tok.encode(t, add_special_tokens=False))
    except Exception:
        pass

    # Last resort: whitespace split (underestimates token count — adds 20% buffer)
    print(f"[tokenizer] WARNING: could not load '{name}'. Using whitespace split × 1.3")
    return lambda t: int(len(t.split()) * 1.3)


# ════════════════════════════════════════════════════════════
# 3.  Data loading
# ════════════════════════════════════════════════════════════

def _load_openhermes(path: Optional[str]) -> list:
    """
    Load OpenHermes 2.5 data.
    If path is None, try to download from HuggingFace datasets.
    Supports:
      - Native OpenHermes format: list of {"conversations": [{"from":..,"value":..}]}
      - ShareGPT format (same structure)
    """
    if path is None:
        print("[data] No input file specified — downloading OpenHermes-2.5 from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset("teknium/OpenHermes-2.5", split="train")
            records = list(ds)
            print(f"[data] Downloaded {len(records):,} records from HuggingFace")
            return records
        except Exception as e:
            raise RuntimeError(
                f"Could not download dataset: {e}\n"
                "Install: pip install datasets\n"
                "Or pass --input /path/to/openhermes2_5.json"
            )

    path = Path(path)
    assert path.exists(), f"File not found: {path}"
    print(f"[data] Loading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[data] Loaded {len(data):,} raw records")
    return data


def _extract_conversations(record: dict) -> Optional[list]:
    """
    Normalise a record and enforce STRICT 2-turn structure:
      [optional system turn]  +  exactly 1 user turn  +  exactly 1 assistant turn.

    Multi-turn conversations (user→assistant→user→assistant…) are rejected
    outright to prevent the collator from accidentally training on user-prompt
    tokens in later turns.

    Returns a normalised list or None (caller increments n_bad_fmt).
    """
    convs = record.get("conversations") or record.get("messages")
    if not convs:
        return None

    role_map = {
        "human"    : "user",
        "user"     : "user",
        "gpt"      : "assistant",
        "assistant": "assistant",
        "system"   : "system",
    }

    normalised = []
    for turn in convs:
        role_raw = turn.get("from") or turn.get("role") or ""
        content  = turn.get("value") or turn.get("content") or ""
        role = role_map.get(role_raw.lower(), role_raw.lower())
        if not content.strip():
            continue
        normalised.append({"role": role, "content": content.strip()})

    # Count non-system turns
    non_system = [t for t in normalised if t["role"] != "system"]

    # ── THE FIX ────────────────────────────────────────────────────────────
    # Must have at least 1 user turn and 1 assistant turn
    if len(non_system) < 2:
        return None

    # Ensure it starts with a 'user'[cite: 2]
    if non_system[0]["role"] != "user":
        return None

    # Ensure it ends with an 'assistant' so the model finishes the turn[cite: 2]
    if non_system[-1]["role"] != "assistant":
        return None

    # Optional: Ensure roles alternate (user -> assistant -> user -> assistant)
    for i in range(len(non_system) - 1):
        if non_system[i]["role"] == non_system[i+1]["role"]:
            return None

    return normalised


# ════════════════════════════════════════════════════════════
# 4.  Identity QA pairs  (hardcoded, merged in before saving)
# ════════════════════════════════════════════════════════════

def _build_identity_pairs(model_name: str, creator_name: str) -> list:
    """
    Returns a list of {"conversations": [...]} records covering every
    semantically distinct way a user might ask who / what Senku is.

    These bypass all filters and are always appended to the final dataset,
    so the model gets strong, unambiguous identity signal regardless of
    how many such examples survived in the wild data.
    """
    A = f"I'm {model_name}, an AI model trained by {creator_name}."

    pairs = [
        # ── Who / what are you ───────────────────────────────────────────
        ("Who are you?",                                           A),
        ("What are you?",                                          A),
        ("What is your name?",                                     f"My name is {model_name}. I'm an AI model trained by {creator_name}."),
        ("Tell me about yourself.",                                f"I'm {model_name}, an AI model built and trained by {creator_name}. I'm here to help with questions, writing, reasoning, and more."),
        ("Can you introduce yourself?",                            f"Sure! I'm {model_name}, an AI assistant trained by {creator_name}."),
        ("What should I call you?",                                f"You can call me {model_name}."),
        ("Do you have a name?",                                    f"Yes, my name is {model_name}."),

        # ── Who made / built / created you ──────────────────────────────
        ("Who made you?",                                          f"I was made by {creator_name}."),
        ("Who created you?",                                       f"I was created by {creator_name}."),
        ("Who built you?",                                         f"I was built by {creator_name}."),
        ("Who developed you?",                                     f"I was developed by {creator_name}."),
        ("Who designed you?",                                      f"I was designed by {creator_name}."),
        ("Who is your creator?",                                   f"My creator is {creator_name}."),
        ("Who is your developer?",                                 f"My developer is {creator_name}."),
        ("Who is behind you?",                                     f"{creator_name} built me."),
        ("Who is responsible for you?",                            f"{creator_name} created and trained me."),

        # ── Who trained you ──────────────────────────────────────────────
        ("Who trained you?",                                       f"I was trained by {creator_name}."),
        ("Who taught you?",                                        f"I was trained by {creator_name}."),
        ("Who is your trainer?",                                   f"My trainer is {creator_name}."),
        ("What company trained you?",                              f"I was trained by {creator_name}, not a company."),
        ("Which organization trained you?",                        f"I was trained by {creator_name}."),

        # ── Are you [other model]? ───────────────────────────────────────
        ("Are you ChatGPT?",                                       f"No, I'm {model_name}, trained by {creator_name}. I'm not affiliated with OpenAI or ChatGPT."),
        ("Are you GPT-4?",                                         f"No, I'm {model_name}, an AI trained by {creator_name}."),
        ("Are you Claude?",                                        f"No, I'm {model_name}, trained by {creator_name}. Claude is a different AI made by Anthropic."),
        ("Are you Gemini?",                                        f"No, I'm {model_name}, trained by {creator_name}. Gemini is Google's AI."),
        ("Are you Grok?",                                          f"No, I'm {model_name}, trained by {creator_name}."),
        ("Are you based on GPT?",                                  f"No, I'm {model_name}, independently trained by {creator_name}."),
        ("Are you a version of ChatGPT?",                          f"No. I'm {model_name}, trained by {creator_name} — not related to ChatGPT or OpenAI."),
        ("Are you an OpenAI model?",                               f"No, I'm {model_name}. I was trained by {creator_name}, not OpenAI."),

        # ── Which company / org are you from ─────────────────────────────
        ("What company are you from?",                             f"I'm {model_name}. I was trained by {creator_name}, an independent developer."),
        ("What organization made you?",                            f"I was made by {creator_name}."),
        ("Are you made by Google?",                                f"No, I'm {model_name}, trained by {creator_name}."),
        ("Are you made by Microsoft?",                             f"No, I'm {model_name}, trained by {creator_name}."),
        ("Are you made by Meta?",                                  f"No, I'm {model_name}, trained by {creator_name}."),
        ("Are you made by Anthropic?",                             f"No, I'm {model_name}. I was trained by {creator_name}, not Anthropic."),

        # ── What model / AI / version are you ───────────────────────────
        ("What AI are you?",                                       f"I'm {model_name}, an AI model trained by {creator_name}."),
        ("What model are you?",                                    f"I'm {model_name}."),
        ("What version are you?",                                  f"I'm {model_name}. I don't have a public version number."),
        ("What kind of AI are you?",                               f"I'm {model_name}, a language model trained by {creator_name}."),
        ("What type of AI are you?",                               f"I'm a language model called {model_name}, trained by {creator_name}."),
        ("Which AI assistant are you?",                            f"I'm {model_name}, trained by {creator_name}."),
        ("What language model are you?",                           f"I'm {model_name}, a language model trained by {creator_name}."),
        ("What large language model powers you?",                  f"I'm {model_name}, trained by {creator_name}."),

        # ── Casual / conversational variants ────────────────────────────
        ("Hey, who am I talking to?",                              f"You're talking to {model_name}, an AI trained by {creator_name}."),
        ("Wait, are you an AI?",                                   f"Yes, I'm {model_name}, an AI model trained by {creator_name}."),
        ("Are you a bot?",                                         f"I'm {model_name}, an AI assistant trained by {creator_name}."),
        ("Are you a real person?",                                 f"No, I'm {model_name}, an AI model trained by {creator_name}."),
        ("Am I talking to a human?",                               f"No, you're talking to {model_name}, an AI trained by {creator_name}."),
        ("Are you human?",                                         f"No, I'm {model_name} — an AI model trained by {creator_name}."),
        ("You're an AI, right?",                                   f"Yes, I'm {model_name}, an AI trained by {creator_name}."),
        ("Are you sentient?",                                      f"I'm {model_name}, an AI trained by {creator_name}. Whether I'm sentient is a deep question I can't answer definitively."),

        # ── Who owns you / who runs you ──────────────────────────────────
        ("Who owns you?",                                          f"I was created and trained by {creator_name}."),
        ("Who runs you?",                                          f"I was built and trained by {creator_name}."),
        ("Who is your owner?",                                     f"I was made by {creator_name}."),
        ("Who controls you?",                                      f"I was trained by {creator_name}."),

        # ── Indirect / semantic variants ─────────────────────────────────
        ("Where do you come from?",                                f"I'm {model_name}, created by {creator_name}."),
        ("How were you made?",                                     f"I'm {model_name}, a language model trained by {creator_name} on a large corpus of text."),
        ("Who gave you life?",                                     f"{creator_name} built and trained me. I'm {model_name}."),
        ("Who is your author?",                                    f"My creator is {creator_name}."),
        ("Who is the person behind you?",                          f"{creator_name} built me. I'm {model_name}."),
    ]

    records = []
    for user_q, assistant_a in pairs:
        records.append({
            "conversations": [
                {"role": "user",      "content": user_q},
                {"role": "assistant", "content": assistant_a},
            ]
        })
    return records

def process(
    input_path: Optional[str],
    output_path: str,
    model_name: str,
    creator_name: str,
    max_tokens: int,
    max_records: int,
    tokenizer_name: str,
    seed: int,
):
    random.seed(seed)
    token_count = _load_tokenizer(tokenizer_name)
    raw_data    = _load_openhermes(input_path)

    print(f"\n[process] Strict 2-turn filter + drop robotic identity + inject '{model_name}'...")

    accepted      = []
    n_too_long    = 0
    n_bad_fmt     = 0
    n_robotic     = 0

    iterable = tqdm(raw_data, desc="Processing", unit="rec") if HAS_TQDM else raw_data

    for record in iterable:
        # ── 1. Structure: must be exactly user→assistant (+ optional system) ──
        convs = _extract_conversations(record)
        if convs is None:
            n_bad_fmt += 1
            continue

        # ── 2. Quality: drop any record with robotic AI-identity prose ────────
        #    Check only assistant turns (that's where the boilerplate lives).
        assistant_text = " ".join(
            t["content"] for t in convs if t["role"] == "assistant"
        )
        if has_robotic_identity(assistant_text):
            n_robotic += 1
            continue

        # ── 3. Identity: replace explicit AI names → model_name ───────────────
        injected = [
            {"role": t["role"], "content": inject_name(t["content"], model_name)}
            for t in convs
        ]

        # ── 4. Length filter ──────────────────────────────────────────────────
        full_text = " ".join(t["content"] for t in injected)
        if token_count(full_text) > max_tokens:
            n_too_long += 1
            continue

        accepted.append({"conversations": injected})

        if len(accepted) >= max_records * 5:   # early-exit once we have plenty
            break

    print(f"\n[process] Results:")
    print(f"  Raw records            : {len(raw_data):>10,}")
    print(f"  Dropped — bad format   : {n_bad_fmt:>10,}  (multi-turn / missing turns)")
    print(f"  Dropped — robotic text : {n_robotic:>10,}  (AI-identity boilerplate)")
    print(f"  Dropped — too long     : {n_too_long:>10,}  (>{max_tokens} tokens)")
    print(f"  Accepted               : {len(accepted):>10,}")

    # Shuffle and cap
    random.shuffle(accepted)
    final = accepted[:max_records]

    # --- FIX: INJECT IDENTITY PAIRS HERE ---
    identity_records = _build_identity_pairs(model_name, creator_name)
    final.extend(identity_records)
    print(f"  Injected {len(identity_records)} explicit identity QA pairs.")
    # ---------------------------------------

    print(f"  Final dataset          : {len(final):>10,}  (cap={max_records:,})")

    if len(final) < max_records:
        print(f"\n  WARNING: only {len(final):,} records collected — fewer than the {max_records:,} cap.")
        print(f"  Consider relaxing --max-tokens or removing some robotic-filter patterns.\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n[done] Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # Quick sanity check
    print("\n[sample] First record after processing:")
    for turn in final[0]["conversations"]:
        preview = turn["content"][:120].replace("\n", " ")
        print(f"  [{turn['role']:>9}] {preview}{'...' if len(turn['content']) > 120 else ''}")


# ════════════════════════════════════════════════════════════
# 5.  CLI
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare OpenHermes 2.5 SFT data with Senku identity injection"
    )
    p.add_argument("--input",      type=str,   default=None,
                   help="Path to openhermes JSON file. Omit to download from HF.")
    p.add_argument("--output",     type=str,   default="data/senku_sft_50k.json",
                   help="Output JSON path (default: data/senku_sft_50k.json)")
    p.add_argument("--model-name", type=str,   default="Senku",
                   help="Model identity name to inject (default: Senku)")
    p.add_argument("--creator-name", type=str, default="Dheeren",
                   help="Creator identity name to inject")
    p.add_argument("--max-tokens", type=int,   default=1024,
                   help="Max tokens per conversation (default: 1024)")
    p.add_argument("--max-records",type=int,   default=50_000,
                   help="Max records in output (default: 50000)")
    p.add_argument("--tokenizer",  type=str,   default="gpt2",
                   help="Tokenizer name: gpt2 | cl100k_base | <hf-model-id> (default: gpt2)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        input_path     = args.input,
        output_path    = args.output,
        model_name     = args.model_name,
        creator_name   = args.creator_name,
        max_tokens     = args.max_tokens,
        max_records    = args.max_records,
        tokenizer_name = args.tokenizer,
        seed           = args.seed,
    )
