import os

import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = os.getenv("SAVE_DIR", "checkpoints/tinyllama_mod_v1")


def human_millions(n: int) -> str:
    return f"{n / 1_000_000:.2f}M"


def module_param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def shape_str(t: torch.Tensor) -> str:
    return "x".join(str(d) for d in t.shape)


def main():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model path not found: {MODEL_PATH}\n"
            "Set SAVE_DIR or run load_model.py first."
        )

    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, device_map=None)
    model.eval()

    total_params = module_param_count(model)
    print(f"Total parameters: {total_params} ({human_millions(total_params)})")
    print("----")

    # Llama-style layout: model.embed_tokens, model.layers, model.norm, lm_head
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        if hasattr(base, "embed_tokens"):
            p = module_param_count(base.embed_tokens)
            print(f"embed_tokens: params={p} ({human_millions(p)})")

        for i, layer in enumerate(base.layers):
            p = module_param_count(layer)
            print(f"layer[{i:02d}] {layer.__class__.__name__}: params={p} ({human_millions(p)})")
            for name, param in layer.named_parameters():
                print(
                    f"  - {name}: shape=({shape_str(param)}), "
                    f"numel={param.numel()} ({human_millions(param.numel())})"
                )

        if hasattr(base, "norm"):
            p = module_param_count(base.norm)
            print(f"final_norm: params={p} ({human_millions(p)})")

        if hasattr(model, "lm_head"):
            p = module_param_count(model.lm_head)
            print(f"lm_head: params={p} ({human_millions(p)})")
        return

    # Generic fallback for other model families.
    print("Could not find `model.layers`; printing named children instead.")
    for name, child in model.named_children():
        p = module_param_count(child)
        print(f"{name}: {child.__class__.__name__}, params={p} ({human_millions(p)})")


if __name__ == "__main__":
    main()
