"""
Layer-by-layer M3 k-grid-search evaluation for Mistral.

Purpose:
- keep the model unchanged on disk,
- patch exactly one transformer nn.Linear at a time,
- evaluate that same layer across a grid of M3 k values,
- write the results into a markdown matrix report.

Important design choice:
- We do NOT change the Mistral model class.
- We do NOT change the checkpoint format.
- We do NOT restructure or save modified weights.
- We only replace one selected layer's runtime forward() computation.

This is meant to answer:
- Which transformer linear layers are most sensitive to M3 approximation?
- For each layer, how does sensitivity change as k increases?
"""

import math
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------------
# User-editable settings
# -----------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/mistral_7b_instruct_v03_gptq_nf4_optimized")

DATASET = os.getenv("DATASET", "wikitext")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "wikitext-2-raw-v1")
SPLIT = os.getenv("SPLIT", "test")
TEXT_FIELD = os.getenv("TEXT_FIELD", "text")
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "100"))

STRIDE = int(os.getenv("STRIDE", "256"))
EVAL_MAX_LENGTH = int(os.getenv("EVAL_MAX_LENGTH", "512"))

SEED = int(os.getenv("SEED", "1337"))

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = os.getenv(
    "DTYPE",
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
)

# Grid-search values for one-layer-at-a-time M3 evaluation.
M3_K_VALUES = [
    int(part.strip())
    for part in os.getenv("M3_K_VALUES", "256,512,1024,2048,4096,8192").split(",")
    if part.strip()
]

# This keeps the math identical while reducing peak VRAM by processing tokens in
# smaller chunks inside the patched linear layer.
M3_TOKEN_CHUNK_SIZE = int(os.getenv("M3_TOKEN_CHUNK_SIZE", "128"))

REPORT_MD_PATH = os.getenv(
    "REPORT_MD_PATH",
    "M3_single_layer_sensitivity_mistral_report_quantized.md",
)


# -----------------------------------------------------------------------------
# Reproducibility and mixed precision context
# -----------------------------------------------------------------------------

if SPLIT not in {"train", "validation", "val", "test"}:
    raise ValueError(
        f"Invalid split '{SPLIT}'. Expected one of: train, validation, val, test."
    )

if not M3_K_VALUES:
    raise ValueError("M3_K_VALUES must contain at least one integer k value.")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if DEVICE == "cuda" and not torch.cuda.is_available():
    raise ValueError("Requested DEVICE='cuda' but CUDA is not available.")

DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"
PTDTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[DTYPE]

CTX = nullcontext() if DEVICE_TYPE == "cpu" else torch.amp.autocast(
    device_type=DEVICE_TYPE,
    dtype=PTDTYPE,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_eval_begins(seq_len, stride, max_len):
    """
    Precompute the actual sliding-window start positions.

    This avoids the tqdm total mismatch that happens when max_len > stride and
    the loop reaches the end of the sequence early.
    """
    begins = []
    for begin in range(0, seq_len, stride):
        begins.append(begin)
        end = min(begin + max_len, seq_len)
        if end == seq_len:
            break
    return begins


def print_loss_and_ppl(label, mean_loss):
    ppl = math.exp(mean_loss)
    print(f"{label} loss: {mean_loss:.6f}")
    print(f"{label} ppl : {ppl:.6f}")


def collect_transformer_linear_layers(current_model):
    """
    Collect only transformer block nn.Linear layers.

    For Mistral this means the linears inside model.layers.<i>.* and excludes:
    - embed_tokens
    - layer norms
    - final norm
    - lm_head
    """
    layer_tuples = []
    for block_index, block in enumerate(current_model.model.layers):
        for local_name, module in block.named_modules():
            if not local_name:
                continue
            if isinstance(module, nn.Linear):
                full_path = f"model.layers.{block_index}.{local_name}"
                layer_tuples.append((full_path, module))
    return layer_tuples


def format_float(value):
    return f"{value:.4f}" if value is not None else ""


def write_report(report_path, baseline_loss, layer_infos, results_by_layer):
    baseline_ppl = math.exp(baseline_loss)

    lines = [
        "# M3 Single-Layer Sensitivity Report",
        "",
        "This report applies M3 to exactly one transformer `nn.Linear` layer at a time",
        "and evaluates a grid of `k` values for that one layer.",
        "",
        "## Settings",
        "",
        f"- `MODEL_PATH`: `{MODEL_PATH}`",
        f"- `DATASET`: `{DATASET}/{DATASET_CONFIG}`",
        f"- `SPLIT`: `{SPLIT}`",
        f"- `TEXT_FIELD`: `{TEXT_FIELD}`",
        f"- `MAX_SAMPLES`: `{MAX_SAMPLES}`",
        f"- `STRIDE`: `{STRIDE}`",
        f"- `EVAL_MAX_LENGTH`: `{EVAL_MAX_LENGTH}`",
        f"- `DEVICE`: `{DEVICE}`",
        f"- `DTYPE`: `{DTYPE}`",
        f"- `M3_K_VALUES`: `{M3_K_VALUES}`",
        f"- `M3_TOKEN_CHUNK_SIZE`: `{M3_TOKEN_CHUNK_SIZE}`",
        "",
        "## Perplexity Grid",
        "",
    ]

    header = ["Layer", "baseline"] + [str(k) for k in M3_K_VALUES]
    divider = ["---"] * len(header)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(divider) + " |")

    for layer_path, _layer in layer_infos:
        row = [f"`{layer_path}`", format_float(baseline_ppl)]
        layer_results = results_by_layer.get(layer_path, {})
        for k in M3_K_VALUES:
            value = None
            if k in layer_results:
                value = layer_results[k]["patched_ppl"]
            row.append(format_float(value))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Loss Delta Grid",
            "",
        ]
    )

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(divider) + " |")

    for layer_path, _layer in layer_infos:
        row = [f"`{layer_path}`", "0.0000"]
        layer_results = results_by_layer.get(layer_path, {})
        for k in M3_K_VALUES:
            value = None
            if k in layer_results:
                value = layer_results[k]["loss_delta"]
            row.append(format_float(value))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Layer Metadata",
            "",
            "| Layer | Weight shape | Best k | Best ppl | Worst k | Worst ppl |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for layer_path, layer in layer_infos:
        layer_results = results_by_layer.get(layer_path, {})
        completed = [(k, stats) for k, stats in layer_results.items()]
        if completed:
            best_k, best_stats = min(completed, key=lambda item: item[1]["patched_ppl"])
            worst_k, worst_stats = max(completed, key=lambda item: item[1]["patched_ppl"])
            best_k_str = str(best_k)
            best_ppl_str = format_float(best_stats["patched_ppl"])
            worst_k_str = str(worst_k)
            worst_ppl_str = format_float(worst_stats["patched_ppl"])
        else:
            best_k_str = ""
            best_ppl_str = ""
            worst_k_str = ""
            worst_ppl_str = ""

        lines.append(
            f"| `{layer_path}` | `{tuple(layer.weight.shape)}` | "
            f"{best_k_str} | {best_ppl_str} | {worst_k_str} | {worst_ppl_str} |"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Step 1: checkpoint loading
# -----------------------------------------------------------------------------

print(f"Loading checkpoint: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=PTDTYPE,
    device_map=None,
)
model.eval()
model.to(DEVICE)


# -----------------------------------------------------------------------------
# Step 2: dataset loading
# -----------------------------------------------------------------------------

print(f"Loading dataset: {DATASET}/{DATASET_CONFIG} [{SPLIT}]")
dataset = load_dataset(DATASET, DATASET_CONFIG, split=SPLIT)
if MAX_SAMPLES is not None:
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

texts = [t for t in dataset[TEXT_FIELD] if isinstance(t, str) and t.strip()]
if not texts:
    raise ValueError("Dataset produced no non-empty text rows.")

corpus = "\n\n".join(texts)
encoded = tokenizer(corpus, return_tensors="pt")

seq_len = encoded.input_ids.size(1)
max_len = min(EVAL_MAX_LENGTH, model.config.max_position_embeddings)
eval_begins = build_eval_begins(seq_len=seq_len, stride=STRIDE, max_len=max_len)

if seq_len <= 1:
    raise ValueError("Tokenized dataset is too short for evaluation.")

if STRIDE <= 0:
    raise ValueError(f"STRIDE must be > 0, got {STRIDE}")

if M3_TOKEN_CHUNK_SIZE <= 0:
    raise ValueError(
        f"M3_TOKEN_CHUNK_SIZE must be > 0, got {M3_TOKEN_CHUNK_SIZE}"
    )


@torch.no_grad()
def evaluate_loss(current_model, desc):
    """
    Compute mean next-token loss using deterministic sliding-window scoring.
    """
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    pbar = tqdm(
        eval_begins,
        total=len(eval_begins),
        desc=desc,
        unit="win",
    )

    for begin in pbar:
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end

        input_ids = encoded.input_ids[:, begin:end].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with CTX:
            outputs = current_model(input_ids, labels=target_ids, use_cache=False)
            neg_log_likelihood = outputs.loss

        nll_sum += neg_log_likelihood.item() * trg_len
        n_tokens += trg_len
        prev_end = end
        pbar.set_postfix(tokens_done=n_tokens)

    return nll_sum / n_tokens


# -----------------------------------------------------------------------------
# Step 3: layer discovery
# -----------------------------------------------------------------------------

target_layers = collect_transformer_linear_layers(model)

print("\nTarget layers")
print("-------------")
for layer_path, layer in target_layers:
    print(f"Path   : {layer_path}")
    print(f"Type   : {type(layer).__name__}")
    print(f"Weight : {tuple(layer.weight.shape)}")
    if layer.bias is not None:
        print(f"Bias   : {tuple(layer.bias.shape)}")
    else:
        print("Bias   : None")
    print("")


# -----------------------------------------------------------------------------
# Step 4: M3 approximation
# -----------------------------------------------------------------------------

def m3_approx_linear_chunked(x, weight, bias, E, k, token_chunk_size):
    """
    Approximate a linear layer output using the same M3-style math as the
    existing script, but process flattened tokens in chunks to reduce peak VRAM.
    """
    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-1])

    norm_w = torch.norm(weight, p=2, dim=1, keepdim=True)
    w_proj = weight @ E
    w_sign = torch.where(w_proj >= 0, 1.0, -1.0)

    y_chunks = []
    for start in range(0, x_flat.shape[0], token_chunk_size):
        stop = min(start + token_chunk_size, x_flat.shape[0])
        x_chunk = x_flat[start:stop]

        norm_x = torch.norm(x_chunk, p=2, dim=1, keepdim=True)
        x_proj = x_chunk @ E
        x_sign = torch.where(x_proj >= 0, 1.0, -1.0)

        S = x_sign @ w_sign.t()
        H = (k - S) / 2.0
        theta = (math.pi * H) / k
        cos_theta = torch.cos(theta)

        y_chunk = cos_theta * (norm_x @ norm_w.t())
        y_chunks.append(y_chunk)

    y_flat = torch.cat(y_chunks, dim=0)
    if bias is not None:
        y_flat = y_flat + bias

    y = y_flat.reshape(*original_shape[:-1], weight.shape[0])
    return y


# -----------------------------------------------------------------------------
# Step 5: single-layer patch helpers
# -----------------------------------------------------------------------------

def patch_linear_layer_with_m3(layer, k, seed_offset):
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    generator = torch.Generator(device=layer.weight.device)
    generator.manual_seed(SEED + seed_offset)

    in_features = layer.weight.shape[1]
    E = torch.randn(
        in_features,
        k,
        generator=generator,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )

    original_forward = layer.forward

    def m3_forward(x):
        return m3_approx_linear_chunked(
            x=x,
            weight=layer.weight,
            bias=layer.bias,
            E=E,
            k=k,
            token_chunk_size=M3_TOKEN_CHUNK_SIZE,
        )

    layer._m3_original_forward = original_forward
    layer._m3_E = E
    layer._m3_k = k
    layer.forward = m3_forward


def restore_original_forward(layer):
    if hasattr(layer, "_m3_original_forward"):
        layer.forward = layer._m3_original_forward
        del layer._m3_original_forward
    if hasattr(layer, "_m3_E"):
        del layer._m3_E
    if hasattr(layer, "_m3_k"):
        del layer._m3_k


# -----------------------------------------------------------------------------
# Step 6: baseline evaluation
# -----------------------------------------------------------------------------

print("\nEvaluating baseline model...")
baseline_loss = evaluate_loss(model, desc="Baseline eval")
baseline_ppl = math.exp(baseline_loss)

print("\nBaseline results")
print("----------------")
print_loss_and_ppl("Baseline", baseline_loss)


# -----------------------------------------------------------------------------
# Step 7: evaluate one layer across the k grid
# -----------------------------------------------------------------------------

results_by_layer = {layer_path: {} for layer_path, _layer in target_layers}
write_report(REPORT_MD_PATH, baseline_loss, target_layers, results_by_layer)
print(f"\nReport initialized: {REPORT_MD_PATH}")

grid_items = [
    (layer_index, layer_path, layer, k)
    for layer_index, (layer_path, layer) in enumerate(target_layers)
    for k in M3_K_VALUES
]

grid_pbar = tqdm(
    grid_items,
    total=len(grid_items),
    desc="Single-layer M3 grid",
    unit="trial",
)

for layer_index, layer_path, layer, k in grid_pbar:
    grid_pbar.set_postfix_str(f"{layer_path} | k={k}")
    print(f"\nPatching one layer with M3: {layer_path} (k={k})")

    patch_linear_layer_with_m3(
        layer,
        k=k,
        seed_offset=layer_index * 100_000 + k,
    )

    try:
        patched_loss = evaluate_loss(
            model,
            desc=f"Eval layer {layer_index + 1}/{len(target_layers)} | k={k}",
        )
        patched_ppl = math.exp(patched_loss)

        results_by_layer[layer_path][k] = {
            "patched_loss": patched_loss,
            "patched_ppl": patched_ppl,
            "loss_delta": patched_loss - baseline_loss,
            "ppl_ratio": patched_ppl / baseline_ppl,
        }

        print_loss_and_ppl("Patched", patched_loss)
        print(f"Loss delta: {patched_loss - baseline_loss:.6f}")
        print(f"PPL ratio : {patched_ppl / baseline_ppl:.6f}")
    finally:
        restore_original_forward(layer)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    write_report(REPORT_MD_PATH, baseline_loss, target_layers, results_by_layer)
    print(f"Updated report: {REPORT_MD_PATH}")


# -----------------------------------------------------------------------------
# Step 8: final summary
# -----------------------------------------------------------------------------

print("\nWorst layer/k combinations by loss delta")
print("----------------------------------------")

flat_results = []
for layer_path, layer_results in results_by_layer.items():
    for k, stats in layer_results.items():
        flat_results.append((layer_path, k, stats))

for rank, (layer_path, k, stats) in enumerate(
    sorted(flat_results, key=lambda item: item[2]["loss_delta"], reverse=True)[:10],
    start=1,
):
    print(
        f"{rank:02d}. {layer_path} | k={k} | "
        f"delta={stats['loss_delta']:.6f} | ppl={stats['patched_ppl']:.6f}"
    )

print(f"\nFinished. Full markdown report written to: {REPORT_MD_PATH}")
