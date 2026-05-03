# Quantizing_small_LLM

Local project for loading a Mistral checkpoint, running interactive chat, inspecting layers, and measuring perplexity before/after quantization.

## Current setup

- Runtime: Linux VM with Python virtual environment
- GPU: NVIDIA RTX 3090 (`24GB` VRAM)
- Model family: Mistral 7B
- Default model id: `mistralai/Mistral-7B-Instruct-v0.3`
- Default local checkpoint: `checkpoints/mistral_7b_instruct_v03`

## Environment

Activate your virtual environment and install dependencies:

```bash
source /venv/main/bin/activate
pip install -r requirements.txt
```

Optional token for gated/private models:

```bash
export HF_TOKEN="hf_your_token"
```

## Scripts

- `load_model.py`
  - Loads from local `SAVE_DIR` if present, otherwise downloads from Hugging Face.
  - Current defaults:
    - `MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3`
    - `SAVE_DIR=checkpoints/mistral_7b_instruct_v03`
  - Saves model + tokenizer to `SAVE_DIR`.

- `chat_local.py`
  - Runs interactive chat with local checkpoint from `SAVE_DIR`.
  - Current generation default: `max_new_tokens=512`.

- `inspect_layers.py`
  - Prints per-layer parameter breakdown for the loaded model.

- `perplexity_sliding.py`
  - Deterministic sliding-window perplexity evaluation on Hugging Face datasets.
  - Current defaults:
    - `dataset=wikitext`
    - `dataset-config=wikitext-2-raw-v1`
    - `split=test`
    - `stride=1024`
    - `eval-max-length=2048`
    - `max-samples=1000`
    - `device=auto`

## Typical workflow

1. Download/save checkpoint:

```bash
python load_model.py
```

2. Chat locally:

```bash
python chat_local.py
```

3. Measure baseline perplexity (pre-quantization):

```bash
python perplexity_sliding.py --model checkpoints/mistral_7b_instruct_v03 --device cuda
```

## Baseline result (before quantization)

- Model: `checkpoints/mistral_7b_instruct_v03`
- Dataset: `wikitext/wikitext-2-raw-v1` (`test` split)
- Method: deterministic sliding-window perplexity
- Settings: `stride=1024`, `eval-max-length=2048`, `max-samples=1000`
- Baseline PPL: **5.2349**

## Requirements

- `torch`
- `transformers`
- `accelerate`
- `huggingface_hub`
- `datasets`
- `tqdm`
