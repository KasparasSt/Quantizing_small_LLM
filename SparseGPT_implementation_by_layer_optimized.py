import os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_calibration_batch(
    dataset_name,
    dataset_config,
    split,
    tokenizer,
    block_size,
    batch_size,
    seed,
    max_rows=4000,
):
    """
    Build one calibration batch from a text dataset.

    This is intentionally kept very close to the GPTQ script so the only
    meaningful algorithmic difference is the compression step itself.
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
    Reuse the optimized GPTQ data-collection path:
    capture the inputs for all target layers in one transformer block during a
    single full-model forward pass.
    """
    activations = {layer_name: [] for layer_name in layer_names}
    handles = []

    for layer_name in layer_names:
        target_layer = get_target_layer(model, block_index, layer_name)

        def make_hook(name):
            def hook(module, inputs, output):
                # Keep cached activations on CPU so the 3090 does not spend its
                # VRAM budget on calibration tensors we only need transiently.
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


def prepare_sparsegpt_hessian(X_flat, percdamp):
    """
    Build the Hessian statistics that SparseGPT uses.

    CHANGED vs GPTQ:
    - GPTQ uses this Hessian inverse to choose quantized replacement values.
    - SparseGPT uses the same second-order structure to choose which weights to
      prune and then compensates the remaining weights with OBS-style updates.
    """
    H = X_flat.t() @ X_flat
    dead = torch.diag(H) == 0

    if dead.any():
        H = H.clone()
        H[dead, dead] = 1

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp

    chol = torch.linalg.cholesky(H.float())
    H_inv = torch.cholesky_inverse(chol)
    # CHANGED explicitly for SparseGPT:
    # the original implementation re-factorizes H^{-1} and then uses that upper
    # triangular factor during the blockwise prune-and-compensate sweep.
    H_inv_factor = torch.linalg.cholesky(H_inv, upper=True)
    return H_inv_factor, dead


def build_unstructured_mask_block(W_block, Hinv1, target_sparsity):
    """
    CHANGED vs GPTQ:
    SparseGPT uses a pruning saliency score instead of a quantization search.
    The score follows the paper/repo logic: w^2 / d^2 where d comes from the
    diagonal of the local inverse-Hessian factor.
    """
    if target_sparsity <= 0:
        return torch.zeros_like(W_block, dtype=torch.bool)
    if target_sparsity >= 1:
        return torch.ones_like(W_block, dtype=torch.bool)

    diag_sq = torch.diag(Hinv1).reshape(1, -1).pow(2).clamp_min(1e-12)
    scores = W_block.pow(2) / diag_sq
    flat_scores = scores.flatten()
    prune_count = int(flat_scores.numel() * target_sparsity)

    if prune_count <= 0:
        return torch.zeros_like(W_block, dtype=torch.bool)
    if prune_count >= flat_scores.numel():
        return torch.ones_like(W_block, dtype=torch.bool)

    threshold = torch.sort(flat_scores)[0][prune_count - 1]
    return scores <= threshold


def apply_nm_mask_step(mask_block, W_block, Hinv1, local_col, prunen, prunem):
    """
    CHANGED vs GPTQ:
    instead of selecting the nearest quantization level, choose the `prunen`
    weakest weights inside each `prunem` group so the output follows the
    semi-structured n:m rule used by SparseGPT.
    """
    if prunen <= 0 or prunem <= 0:
        return
    if local_col % prunem != 0:
        return

    group_end = min(local_col + prunem, W_block.shape[1])
    diag_sq = torch.diag(Hinv1)[local_col:group_end].reshape(1, -1).pow(2).clamp_min(1e-12)
    scores = W_block[:, local_col:group_end].pow(2) / diag_sq
    prune_here = min(prunen, scores.shape[1])
    if prune_here <= 0:
        return

    idx = torch.topk(scores, prune_here, dim=1, largest=False).indices
    mask_block.scatter_(1, local_col + idx, True)


def prune_with_hessian_per_row(
    W,
    H_inv_factor,
    sparsity=0.5,
    prunen=0,
    prunem=0,
    block_cols=128,
):
    """
    SparseGPT-style one-shot pruning with compensation.

    CHANGED vs GPTQ:
    - no quantization grid / scale search
    - masked entries are set to zero
    - surviving weights are reconstructed by the same second-order compensation
      sweep that makes SparseGPT much stronger than plain magnitude pruning
    """
    W_work = W.clone().float()
    rows, columns = W_work.shape
    losses = torch.zeros(rows, device=W_work.device)

    for block_start in range(0, columns, block_cols):
        block_end = min(block_start + block_cols, columns)
        count = block_end - block_start

        W1 = W_work[:, block_start:block_end].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = H_inv_factor[block_start:block_end, block_start:block_end]

        if prunen == 0:
            mask1 = build_unstructured_mask_block(
                W_block=W1,
                Hinv1=Hinv1,
                target_sparsity=sparsity,
            )
        else:
            mask1 = torch.zeros_like(W1, dtype=torch.bool)

        for local_col in range(count):
            apply_nm_mask_step(
                mask_block=mask1,
                W_block=W1,
                Hinv1=Hinv1,
                local_col=local_col,
                prunen=prunen,
                prunem=prunem,
            )

            w = W1[:, local_col]
            d = Hinv1[local_col, local_col].clamp_min(1e-12)

            # CHANGED explicitly from the GPTQ file:
            # instead of quantizing this column to a codebook value, SparseGPT
            # prunes the masked weights by hard-zeroing them.
            q = w.clone()
            q[mask1[:, local_col]] = 0

            Q1[:, local_col] = q
            Losses1[:, local_col] = (w - q).pow(2) / (d ** 2)

            # CHANGED explanation:
            # the compensation update is kept, because this is the part that
            # reconstructs the remaining weights after each pruning decision.
            err1 = (w - q) / d
            W1[:, local_col:] -= err1.unsqueeze(1) @ Hinv1[local_col, local_col:].unsqueeze(0)
            Err1[:, local_col] = err1

        W_work[:, block_start:block_end] = Q1
        losses += torch.sum(Losses1, dim=1) / 2

        if block_end < columns:
            W_work[:, block_end:] -= Err1 @ H_inv_factor[block_start:block_end, block_end:]

    return W_work, torch.sum(losses).item()


def compute_layer_sparsity(W):
    total = W.numel()
    zeros = total - torch.count_nonzero(W).item()
    return zeros / total


if __name__ == "__main__":
    DEVICE = os.getenv("DEVICE", "cuda")  # cuda or cpu
    MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/mistral_7b_instruct_v03")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "checkpoints/mistral_7b_instruct_v03_sparsegpt_90")
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

    DATASET = os.getenv("DATASET", "wikitext")
    DATASET_CONFIG = os.getenv("DATASET_CONFIG", "wikitext-2-raw-v1")
    CALIB_SPLIT = os.getenv("CALIB_SPLIT", "train")
    CALIB_MAX_ROWS = int(os.getenv("CALIB_MAX_ROWS", "4000"))
    BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "512"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
    BLOCK_COLS = int(os.getenv("BLOCK_COLS", "128"))
    SPARSITY = float(os.getenv("SPARSITY", "0.9"))
    PRUNEN = int(os.getenv("PRUNEN", "0"))
    PRUNEM = int(os.getenv("PRUNEM", "0"))
    PERCDAMP = float(os.getenv("PERCDAMP", "0.01"))
    SEED = int(os.getenv("SEED", "555"))

    if PRUNEN < 0 or PRUNEM < 0:
        raise ValueError("PRUNEN and PRUNEM must be non-negative.")
    if (PRUNEN == 0) != (PRUNEM == 0):
        raise ValueError("Use PRUNEN=0 and PRUNEM=0 for unstructured pruning, or set both for n:m pruning.")
    if not 0.0 <= SPARSITY <= 1.0:
        raise ValueError("SPARSITY must be in [0, 1].")
    if PRUNEN and PRUNEN > PRUNEM:
        raise ValueError("PRUNEN must be <= PRUNEM.")

    if DEVICE == "cuda":
        model_dtype = torch.float16
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
    print(f"Prune block cols: {BLOCK_COLS}")
    if PRUNEN == 0:
        print(f"Unstructured sparsity target: {SPARSITY:.4f}")
    else:
        print(f"Semi-structured sparsity target: {PRUNEN}:{PRUNEM}")
    print(f"Per-layer Hessian damping: {PERCDAMP}")

    total_targets = num_blocks * len(LAYER_TYPES)
    overall_layer_pbar = tqdm(
        total=total_targets,
        desc="Pruning layers",
        unit="layer",
        position=0,
        dynamic_ncols=True,
    )
    block_pbar = tqdm(
        range(num_blocks),
        total=num_blocks,
        desc="Pruning blocks",
        unit="block",
        position=1,
        dynamic_ncols=True,
    )

    for block_index in block_pbar:
        block_pbar.set_postfix_str("phase=capture")

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
            block_pbar.set_postfix_str(f"phase=prune layer={clean_layer_name}")
            layer_pbar.set_postfix_str(clean_layer_name)
            target_layer = get_target_layer(model, block_index, layer_name)

            X_big = block_activations[layer_name]
            W_orig, _ = get_layer_weights(target_layer)

            x_dim = W_orig.shape[1]
            X_flat = X_big.view(-1, x_dim).float().to(DEVICE)
            W_orig_gpu = W_orig.float().to(DEVICE)

            with torch.no_grad():
                H_inv_factor, dead = prepare_sparsegpt_hessian(
                    X_flat=X_flat,
                    percdamp=PERCDAMP,
                )
                if dead.any():
                    # CHANGED vs GPTQ:
                    # SparseGPT zeroes dead columns before pruning because the
                    # calibration data says those inputs never appeared.
                    W_orig_gpu[:, dead] = 0

                W_pruned, layer_loss = prune_with_hessian_per_row(
                    W=W_orig_gpu,
                    H_inv_factor=H_inv_factor,
                    sparsity=SPARSITY,
                    prunen=PRUNEN,
                    prunem=PRUNEM,
                    block_cols=BLOCK_COLS,
                )

                target_layer.weight.data.copy_(
                    W_pruned.to(device=target_layer.weight.device, dtype=target_layer.weight.dtype)
                )

            layer_sparsity = compute_layer_sparsity(W_pruned)
            layer_pbar.set_postfix_str(
                f"{clean_layer_name} sp={layer_sparsity:.3f} loss={layer_loss:.2f}"
            )

            del X_big, X_flat, W_orig, W_orig_gpu, H_inv_factor, W_pruned
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            overall_layer_pbar.update(1)
            overall_layer_pbar.set_postfix_str(
                f"block={block_index} layer={clean_layer_name} sp={layer_sparsity:.3f}"
            )

        layer_pbar.close()
        del block_activations

    overall_layer_pbar.close()
    block_pbar.close()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"\nSaved pruned model + tokenizer to: {OUTPUT_PATH}")
