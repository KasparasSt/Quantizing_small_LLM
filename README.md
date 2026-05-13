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

## Optimizing GPTQ approach

- Added `GPTQ_implementation_by_layer_optimized.py` as a copy of the original GPTQ-by-layer script.
- First optimization step completed: activation capture is now done once per transformer block instead of once per target layer.
- In simple terms, the optimized script runs one forward pass to collect inputs for all 7 target linear layers inside a block, then reuses those cached activations while quantizing the layers in that block.
- Next optimization step completed: Hessian inversion now uses a Cholesky-based path instead of `torch.inverse(...)`, which keeps the same idea but is usually faster and more stable.
- Next optimization step completed: TF32 is enabled on CUDA for the optimized script, so large matrix multiplications can use faster tensor-core math on the RTX 3090.
- Next optimization step completed: the inner quantization loop now caches small helper tensors that were being rebuilt every column, which reduces repeated setup work without changing the 10-scale-search flow.
- Phase 2 started: the optimized script now processes weight columns in configurable blocks (`BLOCK_COLS`, default `128`) instead of one long flat column loop, while keeping the same GPTQ-style per-column quantization logic inside each block.
- Speed-focused next step completed: the optimized script still uses one scale per row, but now freezes that per-row scale vector for each column block instead of recomputing it for every single column.
- Speed-focused next step completed: the optimized script now precomputes the scaled Hessian coefficients for each column block, so the hottest inner update loop does less repeated slicing and setup work.
- Speed-focused next step completed: the optimized script now quantizes small column chunks together (`CHUNK_COLS`, default `16`) and applies one larger compensation update per chunk instead of mostly nudging future columns one-by-one.
- Work is now continuing in `GPTQ_implementation_by_layer_optimized_V2.py`.
- In V2, `act-order` was added: columns are sorted by Hessian-diagonal importance before GPTQ-style quantization, then restored to the original order before saving.

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

- `M3_multi_layer_eval_mistral.py`
  - Applies the M3 runtime approximation to a chosen set of Mistral transformer linear layers and evaluates the patched model without modifying the checkpoint.

- `M3_single_layer_sensitivity_mistral.py`
  - Applies M3 to exactly one transformer `nn.Linear` at a time across all decoder blocks.
  - Writes a markdown report to `M3_single_layer_sensitivity_mistral_report.md` so layer sensitivity can be ranked by loss/PPL change.

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

- Dataset/eval settings for all rows: `wikitext/wikitext-2-raw-v1` (`test`), `stride=1024`, `eval-max-length=2048`, `max-samples=1000`.
- On the current system the original GPTQ-style quantization takes **2h 20min**
- On the current system the optimized GPTQ NF4 run takes (`CHUNK_COLS = 16`) **10 min**
- Baseline (unquantized): **5.2349**

| Quant type | GPTQ-processed checkpoint PPL | Real int4 checkpoint PPL | Quant type |
| --- | ---: | ---: | --- |
| `fp4_e3m0` (`QUANT_LIST_ID=1`) | **5.8210** | **5.9089** | fp4 |
| `fp4_e2m1` (`QUANT_LIST_ID=2`) | **5.4486** | **5.5768** | fp4 |
| `fp4_e1m2` (`QUANT_LIST_ID=3`) | **5.5810** | **5.7553** | fp4 |
| `fp4_nf4` (`QUANT_LIST_ID=4`) | **5.4196** | **5.4682** | nf4 | 
| optimized GPTQ NF4 | **5.4191** | - | nf4 |
| bitsandbytes direct quantization | - | **5.4015** | fp4 |
| bitsandbytes direct quantization | - | **5.3241** | nf4 |



## Requirements

- `torch`
- `transformers`
- `accelerate`
- `huggingface_hub`
- `datasets`
- `tqdm`
- `bitsandbytes`



## Implemented M3 matrix multiplication approximation

First each individual layer gridsarch was performed.

After that when k is 8192, and if ppl is no more that 7.6 (baseline is 7.5117), tried to implement the simplifications fo all layers with k = 8192, but PPL is 2404 :D. The table: [M3_single_layer_sensitivity_mistral_report.md](M3_single_layer_sensitivity_mistral_report.md). This results in approximation of 119 out of 224 layers. 

```text
baseline  PPL 7.5117
k = 16384 PPL 46.87
k = 32768 PPL 10.467
k = 65536 PPL 8.54
```

The layers:

```python
TARGET_LAYER_PATHS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.k_proj",
    "model.layers.1.mlp.gate_proj",
    "model.layers.1.mlp.up_proj",
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.k_proj",
    "model.layers.2.self_attn.o_proj",
    "model.layers.2.mlp.gate_proj",
    "model.layers.2.mlp.up_proj",
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.k_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.3.mlp.gate_proj",
    "model.layers.4.self_attn.q_proj",
    "model.layers.4.self_attn.k_proj",
    "model.layers.4.self_attn.o_proj",
    "model.layers.5.self_attn.q_proj",
    "model.layers.5.self_attn.k_proj",
    "model.layers.5.self_attn.o_proj",
    "model.layers.5.mlp.gate_proj",
    "model.layers.5.mlp.up_proj",
    "model.layers.6.self_attn.q_proj",
    "model.layers.6.self_attn.k_proj",
    "model.layers.6.self_attn.o_proj",
    "model.layers.6.mlp.gate_proj",
    "model.layers.7.self_attn.q_proj",
    "model.layers.7.self_attn.k_proj",
    "model.layers.7.mlp.gate_proj",
    "model.layers.7.mlp.up_proj",
    "model.layers.8.self_attn.q_proj",
    "model.layers.8.self_attn.k_proj",
    "model.layers.8.self_attn.o_proj",
    "model.layers.8.mlp.gate_proj",
    "model.layers.9.self_attn.q_proj",
    "model.layers.9.self_attn.k_proj",
    "model.layers.9.self_attn.o_proj",
    "model.layers.9.mlp.gate_proj",
    "model.layers.10.self_attn.q_proj",
    "model.layers.10.self_attn.k_proj",
    "model.layers.10.self_attn.v_proj",
    "model.layers.10.mlp.gate_proj",
    "model.layers.11.self_attn.q_proj",
    "model.layers.11.self_attn.k_proj",
    "model.layers.11.mlp.gate_proj",
    "model.layers.12.self_attn.q_proj",
    "model.layers.12.self_attn.k_proj",
    "model.layers.12.mlp.gate_proj",
    "model.layers.13.self_attn.q_proj",
    "model.layers.13.self_attn.k_proj",
    "model.layers.13.mlp.gate_proj",
    "model.layers.14.self_attn.q_proj",
    "model.layers.14.self_attn.k_proj",
    "model.layers.14.mlp.gate_proj",
    "model.layers.14.mlp.up_proj",
    "model.layers.15.self_attn.q_proj",
    "model.layers.15.self_attn.k_proj",
    "model.layers.15.mlp.gate_proj",
    "model.layers.15.mlp.up_proj",
    "model.layers.16.self_attn.q_proj",
    "model.layers.16.self_attn.k_proj",
    "model.layers.16.self_attn.v_proj",
    "model.layers.16.self_attn.o_proj",
    "model.layers.16.mlp.gate_proj",
    "model.layers.17.self_attn.q_proj",
    "model.layers.17.self_attn.k_proj",
    "model.layers.17.self_attn.o_proj",
    "model.layers.17.mlp.gate_proj",
    "model.layers.18.self_attn.q_proj",
    "model.layers.18.self_attn.k_proj",
    "model.layers.19.self_attn.q_proj",
    "model.layers.19.self_attn.k_proj",
    "model.layers.19.self_attn.o_proj",
    "model.layers.20.self_attn.q_proj",
    "model.layers.20.self_attn.k_proj",
    "model.layers.20.self_attn.o_proj",
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
    "model.layers.24.self_attn.v_proj",
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
```


## M3 matrix multiplication approximation on quantized model nf4

For this investigated nf4 quantized model was taken. Same style gridsearch was applied, where we test M3 approximation with k = 256, 521, 1024, 2048, 4096, 8192. If we set the same acceptable PPL threshold of `0.0883` above the baseline, at `k = 8192` 104/224 layers satisfy the condition. Previously with an unquantized model and the same threshold 119 layers satisfied the threshold.

The avergae loss difference of suitable layers for quantized model M3 is 0.00458, which is similar (even slightly better) than without quantization (0.00488).



The layers satsifying the condition:
```python
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
```

Quantized 104 layers M3

```text
baseline  PPL 7.776
k = 16384 PPL 
k = 32768 PPL 
k = 65536 PPL 8.581
```


Unquantized 104 layers M3

```text
baseline  PPL 7.5117
k = 16384 PPL 
k = 32768 PPL 
k = 65536 PPL 8.0918
```

It is evident, that applying M3 to unquantized data, leads to lower PPL as expected. However, the difference is not large, implying M3 can be used on quantized models to further optimize them.