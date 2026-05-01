import os

import torch
import transformers


SAVE_DIR = os.getenv("SAVE_DIR", "checkpoints/tinyllama_mod_v1")
HF_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Keep answers concise and practical.",
)


def pick_runtime():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = "cpu"
    return dtype, device_map


def main():
    if not os.path.isdir(SAVE_DIR):
        raise FileNotFoundError(
            f"Local weights folder not found: {SAVE_DIR}\n"
            "Run load_model.py once to save local weights, or set SAVE_DIR to the correct path."
        )

    dtype, device_map = pick_runtime()
    print(f"Loading local weights from: {SAVE_DIR}")

    pipe = transformers.pipeline(
        "text-generation",
        model=SAVE_DIR,
        token=HF_TOKEN,
        model_kwargs={"dtype": dtype},
        device_map=device_map,
    )

    terminators = [pipe.tokenizer.eos_token_id]
    eot_id = pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None:
        terminators.append(eot_id)

    print("Interactive chat ready. Type 'exit' to quit.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_text:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        outputs = pipe(
            messages,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        print("Model:", outputs[0]["generated_text"][-1]["content"])


if __name__ == "__main__":
    main()
