import argparse
import os
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM


TARGETS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def get_weight(block, name: str) -> torch.Tensor:
    if name == "self_attn.q_proj.weight":
        return block.self_attn.q_proj.weight
    if name == "self_attn.k_proj.weight":
        return block.self_attn.k_proj.weight
    if name == "self_attn.v_proj.weight":
        return block.self_attn.v_proj.weight
    if name == "self_attn.o_proj.weight":
        return block.self_attn.o_proj.weight
    if name == "mlp.gate_proj.weight":
        return block.mlp.gate_proj.weight
    if name == "mlp.up_proj.weight":
        return block.mlp.up_proj.weight
    if name == "mlp.down_proj.weight":
        return block.mlp.down_proj.weight
    raise ValueError(f"Unknown target name: {name}")


def pack_nibbles(codes_u4: torch.Tensor) -> torch.Tensor:
    """
    Pack uint4 codes (0..15) into bytes (two codes per byte).
    codes_u4 shape: [rows, cols]
    returns shape: [rows, ceil(cols/2)] uint8
    """
    rows, cols = codes_u4.shape
    if cols % 2 == 1:
        pad = torch.zeros((rows, 1), dtype=torch.uint8, device=codes_u4.device)
        codes_u4 = torch.cat([codes_u4, pad], dim=1)
        cols += 1

    lo = codes_u4[:, 0:cols:2]
    hi = codes_u4[:, 1:cols:2]
    return lo | (hi << 4)


def unpack_nibbles(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    interleaved = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return interleaved[:, :original_cols]


def quantize_per_row_group_int4(
    w: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric int4 quantization per-row, per-group.
    - codes are stored as uint4 values in [0..15], using signed mapping q = code - 8.
    - scale per (row, group)
    """
    rows, cols = w.shape
    n_groups = (cols + group_size - 1) // group_size

    codes_u4 = torch.empty((rows, cols), dtype=torch.uint8)
    scales = torch.empty((rows, n_groups), dtype=torch.float32)

    qmin, qmax = -8, 7
    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, cols)

        wg = w[:, start:end]
        absmax = wg.abs().amax(dim=1).clamp_min(eps)
        scale = absmax / float(qmax)
        scales[:, g] = scale.float()

        q = torch.round(wg / scale.unsqueeze(1)).clamp(qmin, qmax).to(torch.int16)
        codes_u4[:, start:end] = (q + 8).to(torch.uint8)

    return codes_u4, scales


def dequantize_per_row_group_int4(
    codes_u4: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    rows, cols = codes_u4.shape
    n_groups = scales.shape[1]
    w_hat = torch.empty((rows, cols), dtype=torch.float32)

    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, cols)
        q = codes_u4[:, start:end].to(torch.int16) - 8
        w_hat[:, start:end] = q.float() * scales[:, g].unsqueeze(1)

    return w_hat


def error_stats(w: torch.Tensor, w_hat: torch.Tensor) -> Dict[str, float]:
    diff = w_hat - w
    mse = torch.mean(diff * diff).item()
    mae = torch.mean(diff.abs()).item()
    max_abs = torch.max(diff.abs()).item()
    rel_l2 = (torch.norm(diff) / (torch.norm(w) + 1e-12)).item()
    return {"mse": mse, "mae": mae, "max_abs": max_abs, "rel_l2": rel_l2}


def output_error_stats(w: torch.Tensor, w_hat: torch.Tensor, n_tokens: int = 2048) -> Dict[str, float]:
    """
    Lightweight functional check:
    compare Y = X W^T vs Y_hat = X W_hat^T on random inputs.
    """
    in_dim = w.shape[1]
    x = torch.randn((n_tokens, in_dim), dtype=torch.float32)
    y = x @ w.t()
    y_hat = x @ w_hat.t()

    diff = y_hat - y
    mse = torch.mean(diff * diff).item()
    rel_l2 = (torch.norm(diff) / (torch.norm(y) + 1e-12)).item()
    return {"mse": mse, "rel_l2": rel_l2}


def main():
    parser = argparse.ArgumentParser(description="Quantize one block to int4 codes+scales and validate round-trip.")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "checkpoints/mistral_7b_instruct_v03"))
    parser.add_argument("--block-index", type=int, default=int(os.getenv("BLOCK_INDEX", "0")))
    parser.add_argument("--group-size", type=int, default=int(os.getenv("GROUP_SIZE", "128")))
    parser.add_argument(
        "--save-dir",
        default=os.getenv("SAVE_DIR", "artifacts/int4_probe"),
        help="Directory to save packed codes and scales.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Do not write packed codes/scales to disk; only print stats.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()

    num_blocks = len(model.model.layers)
    if not (0 <= args.block_index < num_blocks):
        raise ValueError(f"--block-index must be in [0, {num_blocks - 1}], got {args.block_index}")
    block = model.model.layers[args.block_index]

    print(f"Inspecting block: {args.block_index}/{num_blocks - 1} | group_size={args.group_size}")
    if not args.skip_save:
        out_dir = os.path.join(args.save_dir, f"block_{args.block_index:02d}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving int4 artifacts to: {out_dir}")
    print("-----")

    with torch.no_grad():
        for name in TARGETS:
            w = get_weight(block, name).detach().float().cpu()
            codes_u4, scales = quantize_per_row_group_int4(w, group_size=args.group_size)
            packed = pack_nibbles(codes_u4)

            # Verify pack/unpack correctness.
            unpacked = unpack_nibbles(packed, w.shape[1])
            if not torch.equal(unpacked, codes_u4):
                raise RuntimeError(f"Pack/unpack mismatch in {name}")

            w_hat = dequantize_per_row_group_int4(codes_u4, scales, group_size=args.group_size)

            w_stats = error_stats(w, w_hat)
            y_stats = output_error_stats(w, w_hat)
            unique_codes = int(torch.unique(codes_u4).numel())

            print(name)
            print(
                f"  shape={tuple(w.shape)} | unique_codes={unique_codes}/16 | "
                f"packed_bytes={packed.numel()}"
            )
            print(
                f"  weight_err: mse={w_stats['mse']:.6e}, mae={w_stats['mae']:.6e}, "
                f"max_abs={w_stats['max_abs']:.6e}, rel_l2={w_stats['rel_l2']:.6e}"
            )
            print(
                f"  output_err: mse={y_stats['mse']:.6e}, rel_l2={y_stats['rel_l2']:.6e}"
            )
            print("-----")

            if not args.skip_save:
                stem = name.replace(".", "_")
                torch.save(packed.cpu(), os.path.join(out_dir, f"{stem}.codes_u4_packed.pt"))
                torch.save(scales.cpu(), os.path.join(out_dir, f"{stem}.scales.pt"))


if __name__ == "__main__":
    main()
