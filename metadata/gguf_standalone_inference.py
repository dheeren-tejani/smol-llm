import os
import time
import argparse
from llama_cpp import Llama

def main():
    parser = argparse.ArgumentParser(description="Local GGUF Chat Interface")
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf file")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads to use")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="Layers to offload to GPU (0 = CPU only, -1 = all)")
    parser.add_argument("--ctx", type=int, default=1024, help="Context window size")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Loading {args.model}...")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
        type_k=2,
        type_v=2,
        verbose=False,
        flash_attn=True
    )
    print("Model loaded successfully.\n")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("=" * 60)
    print("  Local SLM Chat (Commands: /exit, /reset)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            print("  [Context cleared]")
            continue

        messages.append({"role": "user", "content": user_input})
        print("Assistant> ", end="", flush=True)

        t0 = time.perf_counter()
        token_count = 0
        full_reply = ""

        # Stream generation using the embedded Jinja chat template
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                full_reply += content
                token_count += 1

        t1 = time.perf_counter()
        elapsed = t1 - t0
        tok_per_sec = token_count / elapsed if elapsed > 0 else 0.0

        print(f"\n  [{token_count} tokens | {tok_per_sec:.1f} tok/s]")
        messages.append({"role": "assistant", "content": full_reply})

if __name__ == "__main__":
    main()