import os

import torch
import transformers


MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_TOKEN = os.getenv("HF_TOKEN")

if torch.cuda.is_available():
    # Prefer bf16 on supported GPUs; otherwise use fp16 on CUDA.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_map = "auto"
else:
    dtype = torch.float32
    device_map = "cpu"

if MODEL_ID.startswith("meta-llama/") and not HF_TOKEN:
    print(
        "Note: Meta Llama models usually require Hugging Face access + token. "
        "Set HF_TOKEN in your environment if loading fails with 401/403."
    )

pipe = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    token=HF_TOKEN,
    model_kwargs={"torch_dtype": dtype},
    device_map=device_map,
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

terminators = [pipe.tokenizer.eos_token_id]
eot_id = pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
if eot_id is not None:
    terminators.append(eot_id)

outputs = pipe(
    messages,
    max_new_tokens=128,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print(outputs[0]["generated_text"][-1]["content"])
