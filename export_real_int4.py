import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a model checkpoint to a real bitsandbytes int4 HF checkpoint."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_PATH", "checkpoints/mistral_7b_instruct_v03_gptq_e3m0"),
        help="Source model directory or HF model id.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("OUTPUT_PATH", "checkpoints/mistral_7b_instruct_v03_e3m0_int4"),
        help="Output directory for the int4 checkpoint.",
    )
    parser.add_argument(
        "--quant-type",
        choices=["nf4", "fp4"],
        default=os.getenv("INT4_QUANT_TYPE", "fp4"),
        help="4-bit quant type.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["float16", "bfloat16", "float32"],
        default=os.getenv("INT4_COMPUTE_DTYPE", "float16"),
        help="Compute dtype used by bitsandbytes kernels.",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        help="Enable nested quantization (bnb_4bit_use_double_quant=True).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Optional HF token for gated/private models.",
    )
    return parser.parse_args()


def str_to_torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = parse_args()
    compute_dtype = str_to_torch_dtype(args.compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.double_quant,
    )

    print(f"Loading source model: {args.model}")
    print(
        f"Quant config: load_in_4bit=True, quant_type={args.quant_type}, "
        f"compute_dtype={args.compute_dtype}, double_quant={args.double_quant}"
    )
    print("This requires bitsandbytes to be installed and a CUDA-capable environment.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=args.token,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving int4 model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("Done.")
    print(f"Saved int4 checkpoint: {args.output}")
    print("You can evaluate perplexity by pointing perplexity_sliding.py --model to this directory.")


if __name__ == "__main__":
    main()
