import math
import os
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = os.getenv("SAVE_DIR", "checkpoints/tinyllama_mod_v1")
DEFAULT_DATASET = "wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"
DEFAULT_SPLIT = "test"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def get_texts(dataset_name: str, dataset_config: str, split: str, text_field: str, max_samples: int | None) -> List[str]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts = [t for t in ds[text_field] if isinstance(t, str) and t.strip()]
    if not texts:
        raise ValueError("Dataset produced no non-empty text rows.")
    return texts


def compute_ppl_sliding_window(model, encoded, stride: int, device: str, eval_max_length: int) -> float:
    seq_len = encoded.input_ids.size(1)
    max_len = min(eval_max_length, model.config.max_position_embeddings)

    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    n_windows = math.ceil(seq_len / stride)
    pbar = tqdm(range(0, seq_len, stride), total=n_windows, desc="PPL windows", unit="win")
    for begin in pbar:
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end

        input_ids = encoded.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.inference_mode():
            outputs = model(input_ids, labels=target_ids, use_cache=False)
            neg_log_likelihood = outputs.loss

        nll_sum += neg_log_likelihood.item() * trg_len
        n_tokens += trg_len
        prev_end = end
        pbar.set_postfix(tokens_done=n_tokens)

        if end == seq_len:
            break

    return math.exp(nll_sum / n_tokens)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute perplexity with deterministic sliding window")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model dir or HF model id")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name")
    parser.add_argument("--dataset-config", default=DEFAULT_CONFIG, help="HF dataset config")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--text-field", default="text", help="Text field name in dataset")
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride")
    parser.add_argument(
        "--eval-max-length",
        type=int,
        default=512,
        help="Evaluation context window (<= model max). Lower this to reduce VRAM usage.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Where to run evaluation. Use cpu if CUDA runs out of memory.",
    )
    parser.add_argument("--max-samples", type=int, default=1000, help="Optional cap for faster eval")
    args = parser.parse_args()

    if args.device == "auto":
        device = pick_device()
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested --device cuda but CUDA is not available.")

    dtype = pick_dtype()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    print(f"Loading dataset: {args.dataset}/{args.dataset_config} [{args.split}]")
    texts = get_texts(args.dataset, args.dataset_config, args.split, args.text_field, args.max_samples)
    corpus = "\n\n".join(texts)

    encoded = tokenizer(corpus, return_tensors="pt")
    ppl = compute_ppl_sliding_window(model, encoded, args.stride, device, args.eval_max_length)

    print("-----")
    print(f"tokens: {encoded.input_ids.size(1)}")
    print(f"stride: {args.stride}")
    print(f"eval_max_length: {args.eval_max_length}")
    print(f"device: {device}")
    print(f"perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
