import os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_calibration_batch(dataset_name, dataset_config, split, tokenizer, block_size, batch_size, seed, max_rows=4000):
    """
    Build one GPTQ calibration batch from a text dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.select(range(min(max_rows, len(ds))))
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
    if not texts:
        raise ValueError("Calibration dataset returned no usable text rows.")

    corpus = "\n\n".join(texts)
    ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]
    if len(ids) <= block_size:
        raise ValueError(f"Tokenized corpus too short ({len(ids)} tokens) for block_size={block_size}.")

    all_ids = np.array(ids, dtype=np.int64)
    ix = torch.randint(len(all_ids) - block_size, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(all_ids[i : i + block_size].copy())
            for i in ix
        ]
    ).long()
    return x


def get_activations(model, x_tokens, target_layer):
    activations = []

    # Catch layer inputs for Hessian construction.
    def hook(module, input, output):
        # Move activations to CPU immediately to keep GPU memory low.
        activations.append(input[0].detach().cpu())

    handle = target_layer.register_forward_hook(hook)
    # print("Capturing activations (forward pass)...")
    with torch.no_grad():
        model(x_tokens)
    handle.remove()

    return torch.cat(activations, dim=0)


def get_layer_weights(target_layer):
    W = target_layer.weight.detach().clone()
    b = None
    if target_layer.bias is not None:
        b = target_layer.bias.detach().clone()
    return W, b


def get_target_layer(model, block_index, layer_name):
    """
    Returns one target linear layer from one Mistral decoder block.
    layer_name must be one of:
    - "self_attn.q_proj" or "self_attn.q_proj.weight"
    - "self_attn.k_proj" or "self_attn.k_proj.weight"
    - "self_attn.v_proj" or "self_attn.v_proj.weight"
    - "self_attn.o_proj" or "self_attn.o_proj.weight"
    - "mlp.gate_proj" or "mlp.gate_proj.weight"
    - "mlp.up_proj" or "mlp.up_proj.weight"
    - "mlp.down_proj" or "mlp.down_proj.weight"
    """
    if layer_name.endswith(".weight"):
        layer_name = layer_name[:-7]

    block = model.model.layers[block_index]
    if layer_name == "self_attn.q_proj":
        return block.self_attn.q_proj
    if layer_name == "self_attn.k_proj":
        return block.self_attn.k_proj
    if layer_name == "self_attn.v_proj":
        return block.self_attn.v_proj
    if layer_name == "self_attn.o_proj":
        return block.self_attn.o_proj
    if layer_name == "mlp.gate_proj":
        return block.mlp.gate_proj
    if layer_name == "mlp.up_proj":
        return block.mlp.up_proj
    if layer_name == "mlp.down_proj":
        return block.mlp.down_proj
    raise ValueError(f"Unknown layer name: {layer_name}")


def quantize_with_hessian_per_row(W, H_inv, scale_multiplier=1.0, which_list=4):
    """
    Symmetric int4 quantization with GPTQ-style compensation.
    W: [n_out, n_in]
    H_inv: [n_in, n_in]
    """
    W_quant = W.clone().float()
    n_out, n_in = W.shape

    if which_list == 1:  # fp4_e3m0
        quant_list = W_quant.new_tensor([
            -16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0,
            0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0
        ])
    elif which_list == 2:  # fp4_e2m1
        quant_list = W_quant.new_tensor([
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        ])
    elif which_list == 3:  # fp4_e1m2
        quant_list = W_quant.new_tensor([
            -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
        ])
    elif which_list == 4:  # fp4_nf4
        quant_list = W_quant.new_tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ])
    else:
        raise ValueError("Select quant_list id in {1,2,3,4}.")

    quant_list_max = quant_list.abs().max()
    # print("Weight compensation running (int4)")

    for i in range(n_in):
        w_col = W_quant[:, i].clone()

        # One scale per output row.
        row_absmax = W_quant.abs().amax(dim=1).clamp_min(1e-8)
        s_vec = (row_absmax / quant_list_max) * scale_multiplier

        u = w_col / s_vec
        idx = torch.argmin((u.unsqueeze(1) - quant_list.unsqueeze(0)).abs(), dim=1)
        q_levels = quant_list[idx]
        w_q = q_levels * s_vec

        # GPTQ compensation to the remaining columns.
        error = w_col - w_q
        if i < n_in - 1:
            update = error.unsqueeze(1) @ (H_inv[i, i + 1:] / H_inv[i, i]).unsqueeze(0)
            W_quant[:, i + 1:] -= update

        W_quant[:, i] = w_q

    return W_quant


def find_optimal_scale(W, H_inv, X_flat, Y_ref, which_list=4):
    best_mse = float("inf")
    best_scale = 1.0
    best_W_q = None

    test_scales = np.linspace(0.7, 1.3, 10)
    # print(f"Scale Grid Search ({len(test_scales)} combinations)")

    for s_mult in test_scales:
        W_q_temp = quantize_with_hessian_per_row(
            W,
            H_inv,
            scale_multiplier=s_mult,
            which_list=which_list,
        )

        with torch.no_grad():
            Y_temp = X_flat @ W_q_temp.t()
            mse = torch.nn.functional.mse_loss(Y_temp, Y_ref).item()

        # print(f"S-Mult: {s_mult:.3f} | MSE: {mse:.6f}")

        if mse < best_mse:
            best_mse = mse
            best_scale = s_mult
            best_W_q = W_q_temp

    # print(f"Best Found -> S: {best_scale:.3f} (MSE: {best_mse:.6f})")
    return best_W_q, best_mse


if __name__ == "__main__":
    # Model
    DEVICE = os.getenv("DEVICE", "cuda")  # cuda or cpu
    MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/mistral_7b_instruct_v03")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "checkpoints/mistral_7b_instruct_v03_gptq_nf4")
    QUANT_LIST_ID = int(os.getenv("QUANT_LIST_ID", "4"))  # 1,2,3,4 from quantize_with_hessian_per_row
    LAYER_TYPES = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    # Kept untouched on purpose: model.embed_tokens, all layer norms, model.norm, lm_head.

    # Tokenization / calibration data
    DATASET = "wikitext"
    DATASET_CONFIG = "wikitext-2-raw-v1"
    CALIB_SPLIT = os.getenv("CALIB_SPLIT", "train")
    CALIB_MAX_ROWS = int(os.getenv("CALIB_MAX_ROWS", "4000"))
    BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "512"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
    SEED = 555

    if DEVICE == "cuda":
        model_dtype = torch.float16
        torch.cuda.empty_cache()
    else:
        model_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=model_dtype,
        device_map=None,
    )
    model.to(DEVICE)
    model.eval()

    num_blocks = len(model.model.layers)
    target_layers = [
        (block_index, layer_name)
        for block_index in range(num_blocks)
        for layer_name in LAYER_TYPES
    ]

    x_tokens = get_calibration_batch(
        dataset_name=DATASET,
        dataset_config=DATASET_CONFIG,
        split=CALIB_SPLIT,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        max_rows=CALIB_MAX_ROWS,
    ).to(DEVICE)
    print(f"Batch shape: {tuple(x_tokens.shape)}")
    print(f"Calibration split: {CALIB_SPLIT} (max_rows={CALIB_MAX_ROWS})")

    total_targets = len(target_layers)
    layer_pbar = tqdm(target_layers, total=total_targets, desc="Quantizing layers", unit="layer")
    for layer_idx, (block_index, layer_name) in enumerate(layer_pbar, start=1):
        # print(f"\n=== [{layer_idx}/{total_targets}] Quantizing block {block_index} / {layer_name} ===")
        layer_pbar.set_postfix_str(f"{block_index}:{layer_name}")
        target_layer = get_target_layer(model, block_index, layer_name)

        # Capture inputs for this specific layer.
        X_big = get_activations(model, x_tokens, target_layer=target_layer)
        # print(f"X big shape: {X_big.shape}")

        W_orig, b_orig = get_layer_weights(target_layer)
        # print(f"W_orig Shape: {W_orig.shape}")
        if b_orig is not None:
            # print(f"Original Bias Shape: {b_orig.shape}")
            pass

        x_dim = W_orig.shape[1]
        # Run GPTQ math on GPU.
        X_flat = X_big.view(-1, x_dim).float().to(DEVICE)
        W_orig_gpu = W_orig.float().to(DEVICE)
        # print(f"X_flat size: {X_flat.shape}")

        with torch.no_grad():
            Y_ref = X_flat @ W_orig_gpu.t()
            # print(f"Shape of Y_ref: {Y_ref.shape}")

            # Hessian approximation.
            H = X_flat.t() @ X_flat
            eps = 0.01 * torch.mean(torch.diag(H))
            H += eps * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            H_inv = torch.inverse(H.float())

            best_W_q, best_mse = find_optimal_scale(
                W_orig_gpu,
                H_inv,
                X_flat,
                Y_ref,
                which_list=QUANT_LIST_ID,
            )

            # print(f"Best mse: {best_mse}")
            # print(f"Weights shape: {best_W_q.shape}")
            # print(f"Weights unique sum: {best_W_q.unique().sum().item()}")

            # Replace only this targeted layer.
            target_layer.weight.data.copy_(
                best_W_q.to(device=target_layer.weight.device, dtype=target_layer.weight.dtype)
            )

        # Release layer-local tensors and clear CUDA cache between layers.
        del X_big, X_flat, W_orig, W_orig_gpu, Y_ref, H, H_inv, best_W_q
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"\nSaved quantized model + tokenizer to: {OUTPUT_PATH}")
