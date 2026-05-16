import os

import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = os.getenv("SAVE_DIR", "checkpoints/mistral_7b_instruct_v03_sparsegpt_90")


def human_millions(n: int) -> str:
    return f"{n / 1_000_000:.2f}M"


def pick_runtime():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = "cpu"
    return dtype, device_map


def tensor_zero_stats(tensor: torch.Tensor) -> tuple[int, int, float]:
    total = tensor.numel()
    zeros = total - torch.count_nonzero(tensor).item()
    fraction = zeros / total if total else 0.0
    return zeros, total, fraction


def bucket_name(param_name: str) -> str:
    if ".self_attn." in param_name:
        return "self_attn"
    if ".mlp." in param_name:
        return "mlp"
    if "embed_tokens" in param_name:
        return "embed_tokens"
    if param_name.endswith(".norm.weight") or ".norm." in param_name:
        return "norm"
    if "lm_head" in param_name:
        return "lm_head"
    return "other"


def main():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model path not found: {MODEL_PATH}\n"
            "Set SAVE_DIR or run load_model.py first."
        )

    dtype, device_map = pick_runtime()
    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    total_zeros = 0
    total_values = 0
    bucket_totals: dict[str, list[int]] = {}

    print("Per-parameter zero ratios:")
    print("----")
    for name, param in model.named_parameters():
        zeros, total, fraction = tensor_zero_stats(param)
        total_zeros += zeros
        total_values += total

        bucket = bucket_name(name)
        if bucket not in bucket_totals:
            bucket_totals[bucket] = [0, 0]
        bucket_totals[bucket][0] += zeros
        bucket_totals[bucket][1] += total

        print(
            f"{name}: zeros={zeros} / {total} "
            f"({fraction * 100:.2f}%)"
        )

    print("----")
    overall = total_zeros / total_values if total_values else 0.0
    print(
        f"Overall: zeros={total_zeros} / {total_values} "
        f"({overall * 100:.2f}%)"
    )
    print(
        f"Overall size: {human_millions(total_values)} values, "
        f"{human_millions(total_zeros)} zeros"
    )

    print("----")
    print("By parameter group:")
    for bucket in sorted(bucket_totals):
        zeros, total = bucket_totals[bucket]
        fraction = zeros / total if total else 0.0
        print(f"{bucket}: zeros={zeros} / {total} ({fraction * 100:.2f}%)")


if __name__ == "__main__":
    main()
