import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from architecture import GPT, ModelConfig
from checkpoint import load_checkpoint

def export_and_verify(ckpt_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running conversion on device: {device}")

    print(f"Loading checkpoint from: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = raw_ckpt.get("config", {})

    # 1. Instantiate custom model in pure float32
    model_cfg = ModelConfig(
        vocab_size=cfg_dict.get("vocab_size", 50304),
        d_model=cfg_dict.get("d_model", 768),
        n_layers=cfg_dict.get("n_layers", 12),
        n_heads=cfg_dict.get("n_heads", 12),
        d_ff=cfg_dict.get("d_ff", 2048),
        max_seq_len=cfg_dict.get("max_seq_len", 1024),
        dropout=0.0,
    )
    custom_model = GPT(model_cfg)
    load_checkpoint(ckpt_path, custom_model, device="cpu")
    custom_model = custom_model.to(device=device, dtype=torch.float32).eval()

    # 2. Build matching Hugging Face LlamaConfig
    hf_config = LlamaConfig(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.d_model,
        intermediate_size=model_cfg.d_ff,
        num_hidden_layers=model_cfg.n_layers,
        num_attention_heads=model_cfg.n_heads,
        num_key_value_heads=model_cfg.n_heads,
        max_position_embeddings=model_cfg.max_seq_len,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
    )

    hf_model = LlamaForCausalLM(hf_config)
    src = custom_model.state_dict()
    dst = {}

    # 3. Direct weight mapping
    dst["model.embed_tokens.weight"] = src["token_embed.weight"]
    dst["model.norm.weight"] = src["ln_final.weight"]
    dst["lm_head.weight"] = src["lm_head.weight"]

    for i in range(model_cfg.n_layers):
        dst[f"model.layers.{i}.input_layernorm.weight"] = src[f"blocks.{i}.ln1.weight"]
        dst[f"model.layers.{i}.post_attention_layernorm.weight"] = src[f"blocks.{i}.ln2.weight"]

        # Chunk fused QKV projection along dim=0
        qkv = src[f"blocks.{i}.attn.qkv_proj.weight"]
        q, k, v = qkv.chunk(3, dim=0)
        dst[f"model.layers.{i}.self_attn.q_proj.weight"] = q
        dst[f"model.layers.{i}.self_attn.k_proj.weight"] = k
        dst[f"model.layers.{i}.self_attn.v_proj.weight"] = v
        dst[f"model.layers.{i}.self_attn.o_proj.weight"] = src[f"blocks.{i}.attn.out_proj.weight"]

        # SwiGLU mapping: w1 -> gate, w3 -> up, w2 -> down
        dst[f"model.layers.{i}.mlp.gate_proj.weight"] = src[f"blocks.{i}.ff.w1.weight"]
        dst[f"model.layers.{i}.mlp.up_proj.weight"] = src[f"blocks.{i}.ff.w3.weight"]
        dst[f"model.layers.{i}.mlp.down_proj.weight"] = src[f"blocks.{i}.ff.w2.weight"]

    hf_model.load_state_dict(dst)
    hf_model = hf_model.to(device=device, dtype=torch.float32).eval()

    # 4. Numerical Parity Check (Deterministic FP32 Comparison)
    print("Running numerical parity assertion in float32...")
    torch.manual_seed(42)
    test_input = torch.randint(0, 1000, (1, 32), dtype=torch.long, device=device)

    with torch.no_grad():
        custom_logits, _ = custom_model(test_input)
        hf_logits = hf_model(test_input).logits

    diff = torch.max(torch.abs(custom_logits - hf_logits)).item()
    print(f"Max absolute logit difference (FP32): {diff:.6e}")
    
    assert diff < 1e-3, f"Parity check failed! Logit diff is too high: {diff}"
    print("✓ Parity assertion passed: HF model matches custom model output.")

    # 5. Convert to target bfloat16 and export to disk
    print(f"Converting HF model to bfloat16 and saving to: {output_dir}")
    hf_model.to(dtype=torch.bfloat16)
    hf_model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)
    print("Export complete.")

if __name__ == "__main__":
    export_and_verify(
        ckpt_path=r"C:\Users\dheer\Coding Programs\Projects\chatbot\backend\models\best_raw.pt",
        output_dir="./smol_llama_124m_hf"
    )