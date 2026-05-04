import os

import torch
import transformers


MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")
SAVE_DIR = os.getenv("SAVE_DIR", "checkpoints/mistral_7b_instruct_v03")
if torch.cuda.is_available():
    # Prefer bf16 on supported GPUs; otherwise use fp16 on CUDA.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_map = "auto"
else:
    dtype = torch.float32
    device_map = "cpu"

if not HF_TOKEN:
    print(
        "Note: Set HF_TOKEN in your environment if loading fails with 401/403 "
        "or if the model requires gated access."
    )

model_source = SAVE_DIR if os.path.isdir(SAVE_DIR) else MODEL_ID
if model_source == SAVE_DIR:
    print(f"Loading local weights from: {SAVE_DIR}")
else:
    print(f"Loading model from Hugging Face: {MODEL_ID}")

pipe = transformers.pipeline(
    "text-generation",
    model=model_source,
    token=HF_TOKEN,
    model_kwargs={"dtype": dtype},
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

os.makedirs(SAVE_DIR, exist_ok=True)
pipe.model.save_pretrained(SAVE_DIR, safe_serialization=True)
pipe.tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved model + tokenizer to: {SAVE_DIR}")


