"""
Simple, heavily commented, multi-layer M3 evaluation for Mistral.

This file is intentionally very close to M3_multi_layer_eval_nanogpt.py.

Main difference:
- instead of loading a nanoGPT checkpoint,
- this script loads a regular Hugging Face Mistral checkpoint,
- and evaluates it with deterministic sliding-window perplexity.

This lets you test questions like:
- "What happens if I apply M3 to the whole first Mistral block?"
- "What happens if I patch only the MLP layers in block 0?"
- "What happens if I patch several specific layers at once?"

Important design choice:
- We still do NOT change the Mistral model class.
- We still do NOT change the checkpoint format.
- We still do NOT restructure the stored weights.
- We only replace the runtime forward() computation of selected layers.

So this remains a simple inference-time simulation experiment.
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

DATASET = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"
TEXT_FIELD = "text"
MAX_SAMPLES = 100

STRIDE = 256
EVAL_MAX_LENGTH = 512

SEED = 1337

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = os.getenv(
    "DTYPE",
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
)


# -------------------------------------------------------------------------
# M3 experiment settings
# -------------------------------------------------------------------------

# Paste any layer paths you want here.
#
# Default below = the entire first transformer block (block 0) in standard
# Hugging Face Mistral naming:
# - attention q, k, v, o projections
# - MLP gate, up, and down projections
#
# You can remove lines or add others.
# Example smaller subsets:
# - only MLP in block 0:
#   "model.layers.0.mlp.gate_proj",
#   "model.layers.0.mlp.up_proj",
#   "model.layers.0.mlp.down_proj",
#
# - only attention in block 0:
#   "model.layers.0.self_attn.q_proj",
#   "model.layers.0.self_attn.k_proj",
#   "model.layers.0.self_attn.v_proj",
#   "model.layers.0.self_attn.o_proj",

# TARGET_LAYER_PATHS = [
#     #"model.layers.0.self_attn.q_proj",
#     # "model.layers.0.self_attn.k_proj",
#     # "model.layers.0.self_attn.v_proj",
#     # "model.layers.0.self_attn.o_proj",
#     # "model.layers.0.mlp.gate_proj",
#     # "model.layers.0.mlp.up_proj",
#     # "model.layers.0.mlp.down_proj",
# ]

# These are for the unquantized model, more layers satisfied the PPL threshold.
# TARGET_LAYER_PATHS = [
#     "model.layers.0.self_attn.q_proj",
#     "model.layers.0.self_attn.k_proj",
#     "model.layers.0.mlp.gate_proj",
#     "model.layers.0.mlp.up_proj",
#     "model.layers.1.self_attn.q_proj",
#     "model.layers.1.self_attn.k_proj",
#     "model.layers.1.mlp.gate_proj",
#     "model.layers.1.mlp.up_proj",
#     "model.layers.2.self_attn.q_proj",
#     "model.layers.2.self_attn.k_proj",
#     "model.layers.2.self_attn.o_proj",
#     "model.layers.2.mlp.gate_proj",
#     "model.layers.2.mlp.up_proj",
#     "model.layers.3.self_attn.q_proj",
#     "model.layers.3.self_attn.k_proj",
#     "model.layers.3.self_attn.o_proj",
#     "model.layers.3.mlp.gate_proj",
#     "model.layers.4.self_attn.q_proj",
#     "model.layers.4.self_attn.k_proj",
#     "model.layers.4.self_attn.o_proj",
#     "model.layers.5.self_attn.q_proj",
#     "model.layers.5.self_attn.k_proj",
#     "model.layers.5.self_attn.o_proj",
#     "model.layers.5.mlp.gate_proj",
#     "model.layers.5.mlp.up_proj",
#     "model.layers.6.self_attn.q_proj",
#     "model.layers.6.self_attn.k_proj",
#     "model.layers.6.self_attn.o_proj",
#     "model.layers.6.mlp.gate_proj",
#     "model.layers.7.self_attn.q_proj",
#     "model.layers.7.self_attn.k_proj",
#     "model.layers.7.mlp.gate_proj",
#     "model.layers.7.mlp.up_proj",
#     "model.layers.8.self_attn.q_proj",
#     "model.layers.8.self_attn.k_proj",
#     "model.layers.8.self_attn.o_proj",
#     "model.layers.8.mlp.gate_proj",
#     "model.layers.9.self_attn.q_proj",
#     "model.layers.9.self_attn.k_proj",
#     "model.layers.9.self_attn.o_proj",
#     "model.layers.9.mlp.gate_proj",
#     "model.layers.10.self_attn.q_proj",
#     "model.layers.10.self_attn.k_proj",
#     "model.layers.10.self_attn.v_proj",
#     "model.layers.10.mlp.gate_proj",
#     "model.layers.11.self_attn.q_proj",
#     "model.layers.11.self_attn.k_proj",
#     "model.layers.11.mlp.gate_proj",
#     "model.layers.12.self_attn.q_proj",
#     "model.layers.12.self_attn.k_proj",
#     "model.layers.12.mlp.gate_proj",
#     "model.layers.13.self_attn.q_proj",
#     "model.layers.13.self_attn.k_proj",
#     "model.layers.13.mlp.gate_proj",
#     "model.layers.14.self_attn.q_proj",
#     "model.layers.14.self_attn.k_proj",
#     "model.layers.14.mlp.gate_proj",
#     "model.layers.14.mlp.up_proj",
#     "model.layers.15.self_attn.q_proj",
#     "model.layers.15.self_attn.k_proj",
#     "model.layers.15.mlp.gate_proj",
#     "model.layers.15.mlp.up_proj",
#     "model.layers.16.self_attn.q_proj",
#     "model.layers.16.self_attn.k_proj",
#     "model.layers.16.self_attn.v_proj",
#     "model.layers.16.self_attn.o_proj",
#     "model.layers.16.mlp.gate_proj",
#     "model.layers.17.self_attn.q_proj",
#     "model.layers.17.self_attn.k_proj",
#     "model.layers.17.self_attn.o_proj",
#     "model.layers.17.mlp.gate_proj",
#     "model.layers.18.self_attn.q_proj",
#     "model.layers.18.self_attn.k_proj",
#     "model.layers.19.self_attn.q_proj",
#     "model.layers.19.self_attn.k_proj",
#     "model.layers.19.self_attn.o_proj",
#     "model.layers.20.self_attn.q_proj",
#     "model.layers.20.self_attn.k_proj",
#     "model.layers.20.self_attn.o_proj",
#     "model.layers.21.self_attn.q_proj",
#     "model.layers.21.self_attn.k_proj",
#     "model.layers.21.self_attn.v_proj",
#     "model.layers.21.self_attn.o_proj",
#     "model.layers.22.self_attn.q_proj",
#     "model.layers.22.self_attn.k_proj",
#     "model.layers.22.self_attn.v_proj",
#     "model.layers.22.self_attn.o_proj",
#     "model.layers.23.self_attn.q_proj",
#     "model.layers.23.self_attn.k_proj",
#     "model.layers.23.self_attn.o_proj",
#     "model.layers.24.self_attn.q_proj",
#     "model.layers.24.self_attn.k_proj",
#     "model.layers.24.self_attn.v_proj",
#     "model.layers.24.self_attn.o_proj",
#     "model.layers.25.self_attn.q_proj",
#     "model.layers.25.self_attn.k_proj",
#     "model.layers.25.self_attn.v_proj",
#     "model.layers.25.self_attn.o_proj",
#     "model.layers.26.self_attn.q_proj",
#     "model.layers.26.self_attn.k_proj",
#     "model.layers.26.self_attn.v_proj",
#     "model.layers.26.self_attn.o_proj",
#     "model.layers.27.self_attn.q_proj",
#     "model.layers.27.self_attn.k_proj",
#     "model.layers.27.self_attn.v_proj",
#     "model.layers.27.self_attn.o_proj",
#     "model.layers.28.self_attn.q_proj",
#     "model.layers.28.self_attn.k_proj",
#     "model.layers.28.self_attn.v_proj",
#     "model.layers.28.self_attn.o_proj",
#     "model.layers.29.self_attn.q_proj",
#     "model.layers.29.self_attn.k_proj",
#     "model.layers.29.self_attn.o_proj",
#     "model.layers.30.self_attn.q_proj",
#     "model.layers.30.self_attn.k_proj",
#     "model.layers.30.self_attn.o_proj",
#     "model.layers.31.self_attn.q_proj",
#     "model.layers.31.self_attn.k_proj",
#     "model.layers.31.self_attn.o_proj",
# ]


TARGET_LAYER_PATHS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.1.self_attn.k_proj",
    "model.layers.1.mlp.gate_proj",
    "model.layers.1.mlp.up_proj",
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.k_proj",
    "model.layers.2.mlp.gate_proj",
    "model.layers.2.mlp.up_proj",
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.k_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.4.self_attn.q_proj",
    "model.layers.4.self_attn.k_proj",
    "model.layers.4.self_attn.o_proj",
    "model.layers.4.mlp.gate_proj",
    "model.layers.5.self_attn.q_proj",
    "model.layers.5.self_attn.k_proj",
    "model.layers.5.self_attn.o_proj",
    "model.layers.5.mlp.gate_proj",
    "model.layers.6.self_attn.q_proj",
    "model.layers.6.self_attn.k_proj",
    "model.layers.6.self_attn.o_proj",
    "model.layers.6.mlp.gate_proj",
    "model.layers.7.self_attn.q_proj",
    "model.layers.7.self_attn.k_proj",
    "model.layers.7.mlp.gate_proj",
    "model.layers.8.self_attn.q_proj",
    "model.layers.8.self_attn.k_proj",
    "model.layers.8.self_attn.o_proj",
    "model.layers.9.self_attn.q_proj",
    "model.layers.9.self_attn.k_proj",
    "model.layers.9.mlp.gate_proj",
    "model.layers.10.self_attn.q_proj",
    "model.layers.10.self_attn.k_proj",
    "model.layers.10.mlp.gate_proj",
    "model.layers.11.self_attn.q_proj",
    "model.layers.11.self_attn.k_proj",
    "model.layers.11.mlp.gate_proj",
    "model.layers.12.self_attn.q_proj",
    "model.layers.12.self_attn.k_proj",
    "model.layers.13.self_attn.q_proj",
    "model.layers.13.self_attn.k_proj",
    "model.layers.13.mlp.gate_proj",
    "model.layers.14.self_attn.q_proj",
    "model.layers.14.self_attn.k_proj",
    "model.layers.14.mlp.up_proj",
    "model.layers.15.self_attn.q_proj",
    "model.layers.15.self_attn.k_proj",
    "model.layers.15.mlp.gate_proj",
    "model.layers.16.self_attn.q_proj",
    "model.layers.16.self_attn.k_proj",
    "model.layers.16.mlp.gate_proj",
    "model.layers.17.self_attn.q_proj",
    "model.layers.17.self_attn.k_proj",
    "model.layers.17.self_attn.o_proj",
    "model.layers.18.self_attn.q_proj",
    "model.layers.18.self_attn.k_proj",
    "model.layers.19.self_attn.q_proj",
    "model.layers.19.self_attn.k_proj",
    "model.layers.19.self_attn.o_proj",
    "model.layers.20.self_attn.q_proj",
    "model.layers.20.self_attn.k_proj",
    "model.layers.21.self_attn.q_proj",
    "model.layers.21.self_attn.k_proj",
    "model.layers.21.self_attn.v_proj",
    "model.layers.21.self_attn.o_proj",
    "model.layers.22.self_attn.q_proj",
    "model.layers.22.self_attn.k_proj",
    "model.layers.22.self_attn.v_proj",
    "model.layers.22.self_attn.o_proj",
    "model.layers.23.self_attn.q_proj",
    "model.layers.23.self_attn.k_proj",
    "model.layers.23.self_attn.o_proj",
    "model.layers.24.self_attn.q_proj",
    "model.layers.24.self_attn.k_proj",
    "model.layers.24.self_attn.o_proj",
    "model.layers.25.self_attn.q_proj",
    "model.layers.25.self_attn.k_proj",
    "model.layers.25.self_attn.v_proj",
    "model.layers.25.self_attn.o_proj",
    "model.layers.26.self_attn.q_proj",
    "model.layers.26.self_attn.k_proj",
    "model.layers.26.self_attn.v_proj",
    "model.layers.26.self_attn.o_proj",
    "model.layers.27.self_attn.q_proj",
    "model.layers.27.self_attn.k_proj",
    "model.layers.27.self_attn.v_proj",
    "model.layers.27.self_attn.o_proj",
    "model.layers.28.self_attn.q_proj",
    "model.layers.28.self_attn.k_proj",
    "model.layers.28.self_attn.v_proj",
    "model.layers.28.self_attn.o_proj",
    "model.layers.29.self_attn.q_proj",
    "model.layers.29.self_attn.k_proj",
    "model.layers.29.self_attn.o_proj",
    "model.layers.30.self_attn.q_proj",
    "model.layers.30.self_attn.k_proj",
    "model.layers.30.self_attn.o_proj",
    "model.layers.31.self_attn.q_proj",
    "model.layers.31.self_attn.k_proj",
    "model.layers.31.self_attn.o_proj",

]


# One shared k for all selected layers.
#
# This keeps the script simple.
# If later you want different k per layer, that can be added, but this version
# keeps one global setting for easier interpretation.
M3_K = 65536

# Optional per-layer override for k.
#
# If a layer path is present here, that value is used instead of the global M3_K.
M3_K_BY_LAYER = {}

# These do not change evaluation settings. They only control how the M3
# approximation is computed internally so peak VRAM stays lower.
M3_TOKEN_CHUNK_SIZE = 128
M3_OUT_CHUNK_SIZE = 1024
M3_STORE_E_ON_CPU = True

RUN_LOGITS_MSE_CHECK = True


# -----------------------------------------------------------------------------
# Reproducibility and mixed precision context
# -----------------------------------------------------------------------------

if SPLIT not in {"train", "validation", "val", "test"}:
    raise ValueError(
        f"Invalid split '{SPLIT}'. Expected one of: train, validation, val, test."
    )

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

if seq_len <= 1:
    raise ValueError("Tokenized dataset is too short for evaluation.")

if STRIDE <= 0:
    raise ValueError(f"STRIDE must be > 0, got {STRIDE}")

if M3_TOKEN_CHUNK_SIZE <= 0:
    raise ValueError(
        f"M3_TOKEN_CHUNK_SIZE must be > 0, got {M3_TOKEN_CHUNK_SIZE}"
    )

if M3_OUT_CHUNK_SIZE <= 0:
    raise ValueError(
        f"M3_OUT_CHUNK_SIZE must be > 0, got {M3_OUT_CHUNK_SIZE}"
    )


@torch.no_grad()
def evaluate_loss(current_model):
    """
    Compute mean next-token loss using deterministic sliding-window scoring.
    """
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    n_windows = math.ceil(seq_len / STRIDE)
    pbar = tqdm(
        range(0, seq_len, STRIDE),
        total=n_windows,
        desc="Eval windows",
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

        if end == seq_len:
            break

    return nll_sum / n_tokens


def print_loss_and_ppl(label, mean_loss):
    ppl = math.exp(mean_loss)
    print(f"{label} loss: {mean_loss:.6f}")
    print(f"{label} ppl : {ppl:.6f}")


# -----------------------------------------------------------------------------
# Step 3: module path resolution
# -----------------------------------------------------------------------------

def get_module_by_path(root_module, module_path):
    """
    Resolve a string path like:
        model.layers.0.mlp.down_proj

    Rules:
    - integer path components index into ModuleList-style containers
    - non-integer path components access attributes
    """
    current = root_module
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


target_layers = []
for layer_path in TARGET_LAYER_PATHS:
    layer = get_module_by_path(model, layer_path)
    if not isinstance(layer, nn.Linear):
        raise TypeError(
            f"Target layer '{layer_path}' is {type(layer)}, not nn.Linear."
        )
    target_layers.append((layer_path, layer))

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
# Step 4: simple M3 approximation
# -----------------------------------------------------------------------------

def m3_approx_linear(x, weight, bias, E, k):
    """
    Approximate a linear layer output using M3-style forward logic.

    This version stays mathematically close to the original implementation, but
    computes in token/output chunks so it can run on a 24 GB GPU without
    changing evaluation settings.
    """
    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-1])

    if E.device != weight.device:
        E_work = E.to(weight.device, non_blocking=True)
    else:
        E_work = E

    y_token_chunks = []
    for token_start in range(0, x_flat.shape[0], M3_TOKEN_CHUNK_SIZE):
        token_end = min(token_start + M3_TOKEN_CHUNK_SIZE, x_flat.shape[0])
        x_chunk = x_flat[token_start:token_end]

        norm_x = torch.norm(x_chunk, p=2, dim=1, keepdim=True)
        x_proj = x_chunk @ E_work
        x_sign = torch.where(x_proj >= 0, 1.0, -1.0)

        y_out_chunks = []
        for out_start in range(0, weight.shape[0], M3_OUT_CHUNK_SIZE):
            out_end = min(out_start + M3_OUT_CHUNK_SIZE, weight.shape[0])
            w_chunk = weight[out_start:out_end]

            norm_w = torch.norm(w_chunk, p=2, dim=1, keepdim=True)
            w_proj = w_chunk @ E_work
            w_sign = torch.where(w_proj >= 0, 1.0, -1.0)

            S = x_sign @ w_sign.t()
            H = (k - S) / 2.0
            theta = (math.pi * H) / k
            cos_theta = torch.cos(theta)

            y_chunk = cos_theta * (norm_x @ norm_w.t())
            if bias is not None:
                y_chunk = y_chunk + bias[out_start:out_end]
            y_out_chunks.append(y_chunk)

        y_token_chunks.append(torch.cat(y_out_chunks, dim=1))

    y_flat = torch.cat(y_token_chunks, dim=0)
    y = y_flat.reshape(*original_shape[:-1], weight.shape[0])
    return y


# -----------------------------------------------------------------------------
# Step 5: patch many selected layers
# -----------------------------------------------------------------------------

def patch_linear_layer_with_m3(layer, k, seed_offset):
    """
    Patch one nn.Linear layer.

    seed_offset exists so that each chosen layer gets a different random E,
    while still keeping the whole experiment reproducible from one base seed.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    e_device = "cpu" if M3_STORE_E_ON_CPU else layer.weight.device
    generator = torch.Generator(device=e_device)
    generator.manual_seed(SEED + seed_offset)

    in_features = layer.weight.shape[1]
    E = torch.randn(
        in_features,
        k,
        generator=generator,
        device=e_device,
        dtype=layer.weight.dtype,
    )

    original_forward = layer.forward

    def m3_forward(x):
        return m3_approx_linear(
            x=x,
            weight=layer.weight,
            bias=layer.bias,
            E=E,
            k=k,
        )

    layer._m3_original_forward = original_forward
    layer._m3_E = E
    layer._m3_k = k
    layer.forward = m3_forward

    return original_forward


def restore_original_forward(layer):
    """
    Restore a previously patched layer.
    """
    if hasattr(layer, "_m3_original_forward"):
        layer.forward = layer._m3_original_forward
        del layer._m3_original_forward
    if hasattr(layer, "_m3_E"):
        del layer._m3_E
    if hasattr(layer, "_m3_k"):
        del layer._m3_k


def patch_selected_layers(layer_tuples, k):
    """
    Patch every selected layer in order.

    layer_tuples:
    - list of (layer_path, layer_module)
    """
    for index, (layer_path, layer) in enumerate(layer_tuples):
        layer_k = M3_K_BY_LAYER.get(layer_path, k)
        patch_linear_layer_with_m3(layer, k=layer_k, seed_offset=index)
        print(f"Patched with M3: {layer_path} (k={layer_k})")


# -----------------------------------------------------------------------------
# Step 6: baseline evaluation
# -----------------------------------------------------------------------------

print("\nEvaluating baseline model...")
baseline_loss = evaluate_loss(model)

print("\nBaseline results")
print("----------------")
print_loss_and_ppl("Baseline", baseline_loss)


# -----------------------------------------------------------------------------
# Step 7: optional one-batch logits MSE
# -----------------------------------------------------------------------------

@torch.no_grad()
def get_one_batch_for_logits_check():
    """
    Use the first evaluation window as a deterministic comparison batch.
    """
    end = min(max_len, seq_len)
    input_ids = encoded.input_ids[:, :end].to(DEVICE)
    target_ids = input_ids.clone()
    return input_ids, target_ids


baseline_logits = None
batch_x = None
batch_y = None

if RUN_LOGITS_MSE_CHECK:
    batch_x, batch_y = get_one_batch_for_logits_check()
    with CTX:
        baseline_logits = model(batch_x, labels=batch_y, use_cache=False).logits


# -----------------------------------------------------------------------------
# Step 8: patch selected layers and evaluate again
# -----------------------------------------------------------------------------

print("\nPatching selected layers with M3...")
patch_selected_layers(target_layers, k=M3_K)

patched_loss = evaluate_loss(model)

print("\nPatched results")
print("---------------")
print_loss_and_ppl("Patched", patched_loss)

print("\nDelta")
print("-----")
print(f"Loss delta: {patched_loss - baseline_loss:.6f}")
print(f"PPL ratio : {math.exp(patched_loss) / math.exp(baseline_loss):.6f}")


# -----------------------------------------------------------------------------
# Step 9: optional logits MSE on one batch
# -----------------------------------------------------------------------------

if RUN_LOGITS_MSE_CHECK:
    with CTX:
        patched_logits = model(batch_x, labels=batch_y, use_cache=False).logits

    logits_mse = torch.nn.functional.mse_loss(
        patched_logits.float(),
        baseline_logits.float(),
    ).item()

    print("\nOne-batch logits comparison")
    print("---------------------------")
    print(f"Logits MSE: {logits_mse:.6f}")


# -----------------------------------------------------------------------------
# Step 10: final notes
# -----------------------------------------------------------------------------

print("\nInterpretation notes")
print("--------------------")
print("1. This script changes only the selected layers' forward computations.")
print("2. The checkpoint itself is unchanged.")
print("3. A large loss increase means the chosen multi-layer patch is disruptive.")
print("4. If results are promising, try adding or removing layers one group at a time.")
print("5. If results are poor, it may mean the approximation is too destructive for this combination.")
print(f"6. Evaluation dataset used here: {DATASET}/{DATASET_CONFIG} [{SPLIT}]")
print(f"7. Number of patched layers: {len(target_layers)}")
