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

- `GPTQ_implementation_fast.py`
  - Fast GPTQ-style layer quantization for Mistral (7 target matrices per block).
  - Current practical runtime on RTX 3090: about `2h20min` for one full run.

- `block_int4_probe.py`
  - Tests int4 pack/depack + dequantization on one block and reports reconstruction/output error.

- `export_real_int4.py`
  - Exports checkpoint to real bitsandbytes int4 HF format.
  - Requires:
    - `pip install -U "bitsandbytes>=0.46.1"`

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

## Quantization Results

- Dataset/eval settings for all rows: `wikitext/wikitext-2-raw-v1` (`test`), `stride=1024`, `eval-max-length=2048`, `max-samples=1000`
- Baseline (unquantized): **5.2349**

| Quant type | GPTQ-processed checkpoint PPL | Real int4 checkpoint PPL | Runtime (quantization) |
| --- | ---: | ---: | --- |
| `fp4_e3m0` (`QUANT_LIST_ID=1`) | - | - |  |
| `fp4_e2m1` (`QUANT_LIST_ID=2`) | - | - |  |
| `fp4_e1m2` (`QUANT_LIST_ID=3`) | - | - |  |
| `fp4_nf4` (`QUANT_LIST_ID=4`) | **5.4196** | **5.4682** | ~2h20min |

## Requirements

- `torch`
- `transformers`
- `accelerate`
- `huggingface_hub`
- `datasets`
- `tqdm`
- `bitsandbytes`
