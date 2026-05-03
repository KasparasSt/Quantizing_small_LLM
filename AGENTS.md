# AI Agent Instructions for Quantizing_small_LLM

## Purpose
This repository contains small local tooling for loading and running a TinyLlama-like model locally, chatting with it, and evaluating perplexity via deterministic sliding-window scoring.

## Primary goals for AI agents
- Help maintain and extend the local model-loading pipeline.
- Keep inference and evaluation code compatible with this repo's current baseline hardware (RTX 3090, 24GB VRAM), with CPU fallback support.
- Preserve the current workflow around saved local checkpoints and `HF_TOKEN` usage.
- Prefer workflows that are practical on 24GB VRAM (for example 7B/8B class models and quantized variants), and clearly separate any heavier requirements.

## Key files
- `README.md` — canonical overview, current model details, environment setup, and usage examples.
- `load_model.py` — loads a model from `checkpoints/` if present, otherwise from Hugging Face; saves weights and tokenizer to `SAVE_DIR`; uses `MODEL_ID`, `SAVE_DIR`, and `HF_TOKEN` environment variables.
- `chat_local.py` — runs interactive chat against a local checkpoint at `SAVE_DIR`; uses a system prompt from `SYSTEM_PROMPT` if set.
- `perplexity_sliding.py` — computes sliding-window perplexity for a model on a Hugging Face dataset; defaults to `wikitext/wikitext-2-raw-v1` and a local TinyLlama checkpoint.

## Environment and runtime notes
- `requirements.txt` lists the runtime dependencies: `torch`, `transformers`, `accelerate`, `huggingface_hub`, `datasets`, and `tqdm`.
- `HF_TOKEN` is required when loading gated Hugging Face models or when HTTP access otherwise fails.
- On CUDA, code prefers `bfloat16` if the GPU supports it, otherwise `float16`. On CPU it uses `float32`.
- `perplexity_sliding.py` supports `--device cpu` as a fallback when CUDA memory is constrained.

## Behavior expectations
- Prefer using existing local checkpoint folders under `checkpoints/` rather than switching to remote model downloads when possible.
- Preserve the project’s practical local focus: favor settings that run reliably on 24GB VRAM; keep higher-memory or multi-GPU paths clearly separated.
- Keep command-line interfaces and environment-variable overrides simple and easy to follow.

## When updating this repo
- Link back to `README.md` for usage and environment details rather than duplicating long setup instructions.
- If adding new scripts or major functionality, document the change both in `README.md` and in this file if it affects the agent workflow.
