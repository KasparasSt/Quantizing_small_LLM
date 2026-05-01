# Quantizing_small_LLM

Small local project for loading a small LLM checkpoint, chatting with it, and evaluating perplexity.

## Current status

- Local device constraints (GTX 1650 4GB VRAM) made full GPTQ runs unbearably  slow.
- GPTQ quantization was moved to Kaggle on `T4` GPU.
- Quantization format: `NF4` (`QUANT_LIST_ID=4` in the GPTQ script/notebook).
- Quantized checkpoint path: `checkpoints/tinyllama_gptq_nf4`.
- Kaggle GPTQ run parameters:
  - `MAX_BLOCKS = None`
  - `DATASET = wikitext`
  - `DATASET_CONFIG = wikitext-2-raw-v1`
  - `CALIB_SPLIT = train`
  - `CALIB_MAX_ROWS = 2000`
  - `BLOCK_SIZE = 256`
  - `BATCH_SIZE = 16`
  - `SEED = 555`
  - Scale search points: `10`
- Quantized layers per decoder block:
  - `self_attn.q_proj`
  - `self_attn.k_proj`
  - `self_attn.v_proj`
  - `self_attn.o_proj`
  - `mlp.gate_proj`
  - `mlp.up_proj`
  - `mlp.down_proj`
- Post-quantization perplexity (WikiText-2 test, same sliding-window settings): **9.4680**
- For reference, unquantized PPL was: **8.8369**

## Current model

- Model id: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Local checkpoint dir: `checkpoints/tinyllama_mod_v1`
- Parameters: `1,100,048,384` (~1.10B)
- Architecture: `LlamaForCausalLM`
- Layers: `22`
- Hidden size: `2048`
- Attention heads: `32`
- KV heads: `4`
- Intermediate size: `5632`
- Vocab size: `32000`
- Max context length: `2048` tokens


## Device used

- Laptop: `Lenovo IdeaPad 5 Pro 16ACH6`
- CPU: `AMD Ryzen 7 5800H`
- RAM: `16GB DDR4 3200`
- GPU used in this project: `NVIDIA GeForce GTX 1650 (4GB VRAM)`
- Cloud device used for GPTQ: `Kaggle NVIDIA T4`

Runtime implications for this setup:

- `1.1B` model runs locally.
- Larger models (for example `8B`) require heavy quantization/offloading and are typically slow.


## Environment setup

```powershell
cd C:\Users\kaspa\projects\Quantizing_small_LLM
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional (token for gated models / API limits):

```powershell
setx HF_TOKEN "hf_your_token"
```

Open a new terminal after `setx`.

## Files

- `load_model.py`
  - Loads model (local checkpoint if it exists, otherwise from Hugging Face).
  - Runs one sample generation.
  - Saves model + tokenizer to `SAVE_DIR`.
- `chat_local.py`
  - Loads local checkpoint and runs interactive chat loop.
- `perplexity_sliding.py`
  - Loads dataset and computes perplexity with deterministic sliding window.

## Save and reuse weights

Run once to download and save local checkpoint:

```powershell
python .\load_model.py
```

Override save/load location if needed:

```powershell
$env:SAVE_DIR="checkpoints\my_v2"
python .\load_model.py
```

## Chat with local checkpoint

```powershell
python .\chat_local.py
```

## Perplexity evaluation (deterministic sliding window)

Default dataset config in script:

- Dataset: `wikitext`
- Config: `wikitext-2-raw-v1`
- Split: `test`

Recommended command:

```powershell
python .\perplexity_sliding.py --model checkpoints/tinyllama_mod_v1 --dataset wikitext --dataset-config wikitext-2-raw-v1 --split test --stride 256 --eval-max-length 512 --max-samples 1000 --device cpu
```

Notes:

- `--device cpu` is safer on 4GB VRAM GPUs for perplexity evaluation.
- `--device cuda` may run out of memory depending on `stride` and `eval-max-length`.

## Current reference result

- Baseline model (`checkpoints/tinyllama_mod_v1`):
- Dataset: `wikitext-2-raw-v1` test split
- Method: deterministic sliding window perplexity
- Reported perplexity: **8.8369**
- NF4 quantized model (`checkpoints/tinyllama_gptq_nf4`):
- Dataset: `wikitext-2-raw-v1` test split
- Method: deterministic sliding window perplexity

## Requirements

Current `requirements.txt`:

- `torch`
- `transformers`
- `accelerate`
- `huggingface_hub`
- `datasets`
- `tqdm`
