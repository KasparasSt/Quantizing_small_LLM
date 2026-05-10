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


def get_block_layer_activations(model, x_tokens, block_index, layer_names):
    """
    First optimization step:
    collect the inputs for all target layers in one transformer block
    during a single full-model forward pass.

    Before this change, the script ran one full forward pass per layer.
    Now it runs one full forward pass per block and saves all 7 layer inputs
    for that block at once.
    """
    activations = {layer_name: [] for layer_name in layer_names}
    handles = []

    for layer_name in layer_names:
        target_layer = get_target_layer(model, block_index, layer_name)

        def make_hook(name):
            def hook(module, inputs, output):
                # Keep the cached activations on CPU so GPU memory stays manageable.
                activations[name].append(inputs[0].detach().cpu())
            return hook

        handles.append(target_layer.register_forward_hook(make_hook(layer_name)))

    try:
        with torch.no_grad():
            model(x_tokens)
    finally:
        for handle in handles:
            handle.remove()

    return {
        layer_name: torch.cat(layer_chunks, dim=0)
        for layer_name, layer_chunks in activations.items()
    }


def quantize_with_hessian_per_row(
    W,
    H_inv,
    scale_multiplier=1.0,
    which_list=4,
    block_cols=128,
    chunk_cols=16,
):
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
    # Next optimization step:
    # cache small tensors that are reused for every column so the inner loop
    # does less repeated setup work.
    quant_list_row = quant_list.unsqueeze(0)
    quant_list_cube = quant_list.view(1, 1, -1)
    h_inv_diag_recip = torch.reciprocal(torch.diag(H_inv))

    # Phase 2 start:
    # walk through the input columns in blocks instead of one long flat loop.
    # The math for each individual column is kept the same, but grouping the
    # work like this makes the implementation closer to blockwise GPTQ.
    for block_start in range(0, n_in, block_cols):
        block_end = min(block_start + block_cols, n_in)
        block_width = block_end - block_start
        trailing_slice = slice(block_start, n_in)
        trailing_view = W_quant[:, trailing_slice]
        block_view = W_quant[:, block_start:block_end]

        # Speed-focused change:
        # keep one scale per row, but freeze that row-scale vector for the
        # whole current column block instead of recomputing it for every
        # individual column.
        row_absmax = W_quant.abs().amax(dim=1).clamp_min(1e-8)
        s_vec = (row_absmax / quant_list_max) * scale_multiplier

        # Next vectorization step:
        # precompute the scaled Hessian coefficients for the whole current
        # column block once, instead of rebuilding the same scaled slices for
        # every column update inside the loop.
        scaled_h_inv_block = (
            H_inv[block_start:block_end, block_start:]
            * h_inv_diag_recip[block_start:block_end].unsqueeze(1)
        )

        # Speed-focused next step:
        # quantize a small chunk of columns together, then apply one larger
        # compensation update to the remaining columns. This is faster than
        # doing a separate tiny update for every single column.
        for chunk_start in range(0, block_width, chunk_cols):
            chunk_end = min(chunk_start + chunk_cols, block_width)
            chunk_view = block_view[:, chunk_start:chunk_end].clone()

            u_chunk = chunk_view / s_vec.unsqueeze(1)
            idx_chunk = torch.argmin((u_chunk.unsqueeze(-1) - quant_list_cube).abs(), dim=2)
            q_levels_chunk = quant_list[idx_chunk]
            w_q_chunk = q_levels_chunk * s_vec.unsqueeze(1)
            error_chunk = chunk_view - w_q_chunk

            if block_start + chunk_start < n_in - 1:
                trailing_view[:, chunk_start:] -= error_chunk @ scaled_h_inv_block[chunk_start:chunk_end, chunk_start:]

            # Overwrite the current chunk with its quantized values after the
            # larger compensation update so these columns stay fixed.
            block_view[:, chunk_start:chunk_end] = w_q_chunk

    return W_quant


def find_optimal_scale(W, H_inv, X_flat, Y_ref, which_list=4, block_cols=128, chunk_cols=16):
    best_mse = float("inf")
    best_scale = 1.0
    best_W_q = None

    test_scales = np.linspace(0.7, 1.3, 10)

    for s_mult in test_scales:
        W_q_temp = quantize_with_hessian_per_row(
            W,
            H_inv,
            scale_multiplier=s_mult,
            which_list=which_list,
            block_cols=block_cols,
            chunk_cols=chunk_cols,
        )

        with torch.no_grad():
            Y_temp = X_flat @ W_q_temp.t()
            mse = torch.nn.functional.mse_loss(Y_temp, Y_ref).item()

        if mse < best_mse:
            best_mse = mse
            best_scale = s_mult
            best_W_q = W_q_temp

    return best_W_q, best_mse


def compute_hessian_inverse(H):
    """
    Third optimization step:
    use a Cholesky-based inverse instead of torch.inverse(H).

    This keeps the same overall GPTQ-style flow, but is usually faster and
    numerically safer for a positive-definite damped Hessian.
    """
    chol = torch.linalg.cholesky(H.float())
    return torch.cholesky_inverse(chol)


if __name__ == "__main__":
    # Model
    DEVICE = os.getenv("DEVICE", "cuda")  # cuda or cpu
    MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/mistral_7b_instruct_v03")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "checkpoints/mistral_7b_instruct_v03_gptq_nf4_optimized")
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
    BLOCK_COLS = int(os.getenv("BLOCK_COLS", "128"))
    CHUNK_COLS = int(os.getenv("CHUNK_COLS", "16"))
    SEED = 555

    if DEVICE == "cuda":
        model_dtype = torch.float16
        # Next optimization step:
        # allow TF32 on Ampere GPUs like the RTX 3090 so large float32 matmuls
        # can use faster tensor-core paths.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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
    print(f"Quant block cols: {BLOCK_COLS}")
    print(f"Quant chunk cols: {CHUNK_COLS}")

    total_targets = num_blocks * len(LAYER_TYPES)
    # Show overall progress across all quantized layers, not just blocks.
    overall_layer_pbar = tqdm(
        total=total_targets,
        desc="Quantizing layers",
        unit="layer",
        position=0,
        dynamic_ncols=True,
    )
    block_pbar = tqdm(
        range(num_blocks),
        total=num_blocks,
        desc="Quantizing blocks",
        unit="block",
        position=1,
        dynamic_ncols=True,
    )
    for block_index in block_pbar:
        block_pbar.set_postfix_str("phase=capture")

        # First optimization step:
        # run one forward pass for this block and cache the inputs of all
        # target linear layers inside it.
        block_activations = get_block_layer_activations(
            model=model,
            x_tokens=x_tokens,
            block_index=block_index,
            layer_names=LAYER_TYPES,
        )

        layer_pbar = tqdm(
            LAYER_TYPES,
            total=len(LAYER_TYPES),
            desc=f"Block {block_index:02d} layers",
            unit="layer",
            position=2,
            leave=False,
            dynamic_ncols=True,
        )
        for layer_name in layer_pbar:
            clean_layer_name = layer_name.replace(".weight", "")
            block_pbar.set_postfix_str(f"phase=quantize layer={clean_layer_name}")
            layer_pbar.set_postfix_str(clean_layer_name)
            target_layer = get_target_layer(model, block_index, layer_name)

            # Reuse the cached activation for this layer instead of running
            # another full-model forward pass.
            X_big = block_activations[layer_name]

            W_orig, b_orig = get_layer_weights(target_layer)
            if b_orig is not None:
                pass

            x_dim = W_orig.shape[1]
            X_flat = X_big.view(-1, x_dim).float().to(DEVICE)
            W_orig_gpu = W_orig.float().to(DEVICE)

            with torch.no_grad():
                Y_ref = X_flat @ W_orig_gpu.t()

                H = X_flat.t() @ X_flat
                eps = 0.01 * torch.mean(torch.diag(H))
                H += eps * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
                # Changed from torch.inverse(H.float()) to a Cholesky-based
                # inverse. Same role in the algorithm, but usually faster.
                H_inv = compute_hessian_inverse(H)

                best_W_q, best_mse = find_optimal_scale(
                    W_orig_gpu,
                    H_inv,
                    X_flat,
                    Y_ref,
                    which_list=QUANT_LIST_ID,
                    block_cols=BLOCK_COLS,
                    chunk_cols=CHUNK_COLS,
                )

                target_layer.weight.data.copy_(
                    best_W_q.to(device=target_layer.weight.device, dtype=target_layer.weight.dtype)
                )

            del X_big, X_flat, W_orig, W_orig_gpu, Y_ref, H, H_inv, best_W_q
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            overall_layer_pbar.update(1)
            overall_layer_pbar.set_postfix_str(f"block={block_index} layer={clean_layer_name}")

        layer_pbar.close()
        del block_activations

    overall_layer_pbar.close()
    block_pbar.close()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"\nSaved quantized model + tokenizer to: {OUTPUT_PATH}")
