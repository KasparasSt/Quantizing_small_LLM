# M3 Single-Layer Sensitivity Report

This report applies M3 to exactly one transformer `nn.Linear` layer at a time
and evaluates a grid of `k` values for that one layer.

## Settings

- `MODEL_PATH`: `checkpoints/mistral_7b_instruct_v03_sparsegpt_30`
- `DATASET`: `wikitext/wikitext-2-raw-v1`
- `SPLIT`: `test`
- `TEXT_FIELD`: `text`
- `MAX_SAMPLES`: `100`
- `STRIDE`: `256`
- `EVAL_MAX_LENGTH`: `512`
- `DEVICE`: `cuda`
- `DTYPE`: `bfloat16`
- `M3_K_VALUES`: `[8192]`
- `M3_TOKEN_CHUNK_SIZE`: `128`

## Perplexity Grid

| Layer | baseline | 8192 |
| --- | --- | --- |
| `model.layers.0.self_attn.q_proj` | 7.6952 | 7.6966 |
| `model.layers.0.self_attn.k_proj` | 7.6952 | 7.6654 |
| `model.layers.0.self_attn.v_proj` | 7.6952 | 377.1837 |
| `model.layers.0.self_attn.o_proj` | 7.6952 | 7.9777 |
| `model.layers.0.mlp.gate_proj` | 7.6952 | 7.6861 |
| `model.layers.0.mlp.up_proj` | 7.6952 | 7.6761 |
| `model.layers.0.mlp.down_proj` | 7.6952 | 16.0618 |
| `model.layers.1.self_attn.q_proj` | 7.6952 | 7.7499 |
| `model.layers.1.self_attn.k_proj` | 7.6952 | 7.7571 |
| `model.layers.1.self_attn.v_proj` | 7.6952 | 16.8308 |
| `model.layers.1.self_attn.o_proj` | 7.6952 | 7.8739 |
| `model.layers.1.mlp.gate_proj` | 7.6952 | 7.7425 |
| `model.layers.1.mlp.up_proj` | 7.6952 | 7.7825 |
| `model.layers.1.mlp.down_proj` | 7.6952 | 1294.8345 |
| `model.layers.2.self_attn.q_proj` | 7.6952 | 7.6950 |
| `model.layers.2.self_attn.k_proj` | 7.6952 | 7.7305 |
| `model.layers.2.self_attn.v_proj` | 7.6952 | 8.1298 |
| `model.layers.2.self_attn.o_proj` | 7.6952 | 7.7736 |
| `model.layers.2.mlp.gate_proj` | 7.6952 | 7.8029 |
| `model.layers.2.mlp.up_proj` | 7.6952 | 7.7647 |
| `model.layers.2.mlp.down_proj` | 7.6952 | 8.9304 |
| `model.layers.3.self_attn.q_proj` | 7.6952 | 7.7184 |
| `model.layers.3.self_attn.k_proj` | 7.6952 | 7.7047 |
| `model.layers.3.self_attn.v_proj` | 7.6952 | 8.4296 |
| `model.layers.3.self_attn.o_proj` | 7.6952 | 7.7233 |
| `model.layers.3.mlp.gate_proj` | 7.6952 | 7.7865 |
| `model.layers.3.mlp.up_proj` | 7.6952 | 7.8063 |
| `model.layers.3.mlp.down_proj` | 7.6952 | 8.7045 |
| `model.layers.4.self_attn.q_proj` | 7.6952 | 7.7130 |
| `model.layers.4.self_attn.k_proj` | 7.6952 | 7.7237 |
| `model.layers.4.self_attn.v_proj` | 7.6952 | 8.0945 |
| `model.layers.4.self_attn.o_proj` | 7.6952 | 7.7565 |
| `model.layers.4.mlp.gate_proj` | 7.6952 | 7.8256 |
| `model.layers.4.mlp.up_proj` | 7.6952 | 7.8560 |
| `model.layers.4.mlp.down_proj` | 7.6952 | 8.7192 |
| `model.layers.5.self_attn.q_proj` | 7.6952 | 7.7072 |
| `model.layers.5.self_attn.k_proj` | 7.6952 | 7.7441 |
| `model.layers.5.self_attn.v_proj` | 7.6952 | 7.9900 |
| `model.layers.5.self_attn.o_proj` | 7.6952 | 7.7507 |
| `model.layers.5.mlp.gate_proj` | 7.6952 | 7.7312 |
| `model.layers.5.mlp.up_proj` | 7.6952 | 7.8439 |
| `model.layers.5.mlp.down_proj` | 7.6952 | 8.8567 |
| `model.layers.6.self_attn.q_proj` | 7.6952 | 7.7363 |
| `model.layers.6.self_attn.k_proj` | 7.6952 | 7.7248 |
| `model.layers.6.self_attn.v_proj` | 7.6952 | 7.8808 |
| `model.layers.6.self_attn.o_proj` | 7.6952 | 7.7501 |
| `model.layers.6.mlp.gate_proj` | 7.6952 | 7.7331 |
| `model.layers.6.mlp.up_proj` | 7.6952 | 7.8389 |
| `model.layers.6.mlp.down_proj` | 7.6952 | 8.7921 |
| `model.layers.7.self_attn.q_proj` | 7.6952 | 7.7229 |
| `model.layers.7.self_attn.k_proj` | 7.6952 | 7.7014 |
| `model.layers.7.self_attn.v_proj` | 7.6952 | 7.8467 |
| `model.layers.7.self_attn.o_proj` | 7.6952 | 7.7903 |
| `model.layers.7.mlp.gate_proj` | 7.6952 | 7.7702 |
| `model.layers.7.mlp.up_proj` | 7.6952 | 7.8107 |
| `model.layers.7.mlp.down_proj` | 7.6952 | 8.7455 |
| `model.layers.8.self_attn.q_proj` | 7.6952 | 7.7066 |
| `model.layers.8.self_attn.k_proj` | 7.6952 | 7.7076 |
| `model.layers.8.self_attn.v_proj` | 7.6952 | 7.8178 |
| `model.layers.8.self_attn.o_proj` | 7.6952 | 7.7639 |
| `model.layers.8.mlp.gate_proj` | 7.6952 | 7.7452 |
| `model.layers.8.mlp.up_proj` | 7.6952 | 7.7786 |
| `model.layers.8.mlp.down_proj` | 7.6952 | 8.5205 |
| `model.layers.9.self_attn.q_proj` | 7.6952 | 7.7423 |
| `model.layers.9.self_attn.k_proj` | 7.6952 | 7.7119 |
| `model.layers.9.self_attn.v_proj` | 7.6952 | 7.8289 |
| `model.layers.9.self_attn.o_proj` | 7.6952 | 7.7222 |
| `model.layers.9.mlp.gate_proj` | 7.6952 | 7.7671 |
| `model.layers.9.mlp.up_proj` | 7.6952 | 7.8076 |
| `model.layers.9.mlp.down_proj` | 7.6952 | 8.4275 |
| `model.layers.10.self_attn.q_proj` | 7.6952 | 7.7252 |
| `model.layers.10.self_attn.k_proj` | 7.6952 | 7.7122 |
| `model.layers.10.self_attn.v_proj` | 7.6952 | 7.8284 |
| `model.layers.10.self_attn.o_proj` | 7.6952 | 7.7842 |
| `model.layers.10.mlp.gate_proj` | 7.6952 | 7.7800 |
| `model.layers.10.mlp.up_proj` | 7.6952 | 7.8393 |
| `model.layers.10.mlp.down_proj` | 7.6952 | 8.3285 |
| `model.layers.11.self_attn.q_proj` | 7.6952 | 7.7031 |
| `model.layers.11.self_attn.k_proj` | 7.6952 | 7.7053 |
| `model.layers.11.self_attn.v_proj` | 7.6952 | 7.8481 |
| `model.layers.11.self_attn.o_proj` | 7.6952 | 7.8131 |
| `model.layers.11.mlp.gate_proj` | 7.6952 | 7.7654 |
| `model.layers.11.mlp.up_proj` | 7.6952 | 7.8080 |
| `model.layers.11.mlp.down_proj` | 7.6952 | 8.3204 |
| `model.layers.12.self_attn.q_proj` | 7.6952 | 7.7504 |
| `model.layers.12.self_attn.k_proj` | 7.6952 | 7.7216 |
| `model.layers.12.self_attn.v_proj` | 7.6952 | 7.9365 |
| `model.layers.12.self_attn.o_proj` | 7.6952 | 7.8142 |
| `model.layers.12.mlp.gate_proj` | 7.6952 | 7.7723 |
| `model.layers.12.mlp.up_proj` | 7.6952 | 7.8165 |
| `model.layers.12.mlp.down_proj` | 7.6952 | 8.3285 |
| `model.layers.13.self_attn.q_proj` | 7.6952 | 7.7364 |
| `model.layers.13.self_attn.k_proj` | 7.6952 | 7.6985 |
| `model.layers.13.self_attn.v_proj` | 7.6952 | 7.8139 |
| `model.layers.13.self_attn.o_proj` | 7.6952 | 7.8060 |
| `model.layers.13.mlp.gate_proj` | 7.6952 | 7.7694 |
| `model.layers.13.mlp.up_proj` | 7.6952 | 7.7671 |
| `model.layers.13.mlp.down_proj` | 7.6952 | 8.1962 |
| `model.layers.14.self_attn.q_proj` | 7.6952 | 7.7301 |
| `model.layers.14.self_attn.k_proj` | 7.6952 | 7.7028 |
| `model.layers.14.self_attn.v_proj` | 7.6952 | 7.9326 |
| `model.layers.14.self_attn.o_proj` | 7.6952 | 7.7909 |
| `model.layers.14.mlp.gate_proj` | 7.6952 | 7.7697 |
| `model.layers.14.mlp.up_proj` | 7.6952 | 7.7472 |
| `model.layers.14.mlp.down_proj` | 7.6952 | 8.2392 |
| `model.layers.15.self_attn.q_proj` | 7.6952 | 7.7114 |
| `model.layers.15.self_attn.k_proj` | 7.6952 | 7.7241 |
| `model.layers.15.self_attn.v_proj` | 7.6952 | 7.9205 |
| `model.layers.15.self_attn.o_proj` | 7.6952 | 7.8400 |
| `model.layers.15.mlp.gate_proj` | 7.6952 | 7.7659 |
| `model.layers.15.mlp.up_proj` | 7.6952 | 7.7669 |
| `model.layers.15.mlp.down_proj` | 7.6952 | 8.2970 |
| `model.layers.16.self_attn.q_proj` | 7.6952 | 7.7301 |
| `model.layers.16.self_attn.k_proj` | 7.6952 | 7.7184 |
| `model.layers.16.self_attn.v_proj` | 7.6952 | 7.7866 |
| `model.layers.16.self_attn.o_proj` | 7.6952 | 7.8000 |
| `model.layers.16.mlp.gate_proj` | 7.6952 | 7.7746 |
| `model.layers.16.mlp.up_proj` | 7.6952 | 7.8435 |
| `model.layers.16.mlp.down_proj` | 7.6952 | 8.4903 |
| `model.layers.17.self_attn.q_proj` | 7.6952 | 7.7181 |
| `model.layers.17.self_attn.k_proj` | 7.6952 | 7.7139 |
| `model.layers.17.self_attn.v_proj` | 7.6952 | 7.8803 |
| `model.layers.17.self_attn.o_proj` | 7.6952 | 7.7387 |
| `model.layers.17.mlp.gate_proj` | 7.6952 | 7.7884 |
| `model.layers.17.mlp.up_proj` | 7.6952 | 7.8499 |
| `model.layers.17.mlp.down_proj` | 7.6952 | 8.6974 |
| `model.layers.18.self_attn.q_proj` | 7.6952 | 7.6973 |
| `model.layers.18.self_attn.k_proj` | 7.6952 | 7.7241 |
| `model.layers.18.self_attn.v_proj` | 7.6952 | 7.8256 |
| `model.layers.18.self_attn.o_proj` | 7.6952 | 7.8003 |
| `model.layers.18.mlp.gate_proj` | 7.6952 | 7.8361 |
| `model.layers.18.mlp.up_proj` | 7.6952 | 7.9223 |
| `model.layers.18.mlp.down_proj` | 7.6952 | 9.3264 |
| `model.layers.19.self_attn.q_proj` | 7.6952 | 7.7454 |
| `model.layers.19.self_attn.k_proj` | 7.6952 | 7.7341 |
| `model.layers.19.self_attn.v_proj` | 7.6952 | 7.8805 |
| `model.layers.19.self_attn.o_proj` | 7.6952 | 7.7830 |
| `model.layers.19.mlp.gate_proj` | 7.6952 | 7.8638 |
| `model.layers.19.mlp.up_proj` | 7.6952 | 7.8377 |
| `model.layers.19.mlp.down_proj` | 7.6952 | 9.6844 |
| `model.layers.20.self_attn.q_proj` | 7.6952 | 7.7263 |
| `model.layers.20.self_attn.k_proj` | 7.6952 | 7.7084 |
| `model.layers.20.self_attn.v_proj` | 7.6952 | 7.8232 |
| `model.layers.20.self_attn.o_proj` | 7.6952 | 7.7951 |
| `model.layers.20.mlp.gate_proj` | 7.6952 | 7.8802 |
| `model.layers.20.mlp.up_proj` | 7.6952 | 7.9218 |
| `model.layers.20.mlp.down_proj` | 7.6952 | 9.3816 |
| `model.layers.21.self_attn.q_proj` | 7.6952 | 7.7091 |
| `model.layers.21.self_attn.k_proj` | 7.6952 | 7.7076 |
| `model.layers.21.self_attn.v_proj` | 7.6952 | 7.7771 |
| `model.layers.21.self_attn.o_proj` | 7.6952 | 7.7421 |
| `model.layers.21.mlp.gate_proj` | 7.6952 | 7.8819 |
| `model.layers.21.mlp.up_proj` | 7.6952 | 7.9025 |
| `model.layers.21.mlp.down_proj` | 7.6952 | 9.2029 |
| `model.layers.22.self_attn.q_proj` | 7.6952 | 7.7206 |
| `model.layers.22.self_attn.k_proj` | 7.6952 | 7.7026 |
| `model.layers.22.self_attn.v_proj` | 7.6952 | 7.7663 |
| `model.layers.22.self_attn.o_proj` | 7.6952 | 7.7334 |
| `model.layers.22.mlp.gate_proj` | 7.6952 | 7.8607 |
| `model.layers.22.mlp.up_proj` | 7.6952 | 7.8833 |
| `model.layers.22.mlp.down_proj` | 7.6952 | 9.0134 |
| `model.layers.23.self_attn.q_proj` | 7.6952 | 7.7198 |
| `model.layers.23.self_attn.k_proj` | 7.6952 | 7.6997 |
| `model.layers.23.self_attn.v_proj` | 7.6952 | 7.7939 |
| `model.layers.23.self_attn.o_proj` | 7.6952 | 7.7214 |
| `model.layers.23.mlp.gate_proj` | 7.6952 | 7.9102 |
| `model.layers.23.mlp.up_proj` | 7.6952 | 7.8482 |
| `model.layers.23.mlp.down_proj` | 7.6952 | 8.8605 |
| `model.layers.24.self_attn.q_proj` | 7.6952 | 7.7178 |
| `model.layers.24.self_attn.k_proj` | 7.6952 | 7.6988 |
| `model.layers.24.self_attn.v_proj` | 7.6952 | 7.7641 |
| `model.layers.24.self_attn.o_proj` | 7.6952 | 7.7282 |
| `model.layers.24.mlp.gate_proj` | 7.6952 | 7.9310 |
| `model.layers.24.mlp.up_proj` | 7.6952 | 7.8753 |
| `model.layers.24.mlp.down_proj` | 7.6952 | 8.7923 |
| `model.layers.25.self_attn.q_proj` | 7.6952 | 7.7090 |
| `model.layers.25.self_attn.k_proj` | 7.6952 | 7.7114 |
| `model.layers.25.self_attn.v_proj` | 7.6952 | 7.7587 |
| `model.layers.25.self_attn.o_proj` | 7.6952 | 7.7261 |
| `model.layers.25.mlp.gate_proj` | 7.6952 | 8.0066 |
| `model.layers.25.mlp.up_proj` | 7.6952 | 7.9310 |
| `model.layers.25.mlp.down_proj` | 7.6952 | 8.7561 |
| `model.layers.26.self_attn.q_proj` | 7.6952 | 7.7299 |
| `model.layers.26.self_attn.k_proj` | 7.6952 | 7.7032 |
| `model.layers.26.self_attn.v_proj` | 7.6952 | 7.7740 |
| `model.layers.26.self_attn.o_proj` | 7.6952 | 7.7161 |
| `model.layers.26.mlp.gate_proj` | 7.6952 | 7.9847 |
| `model.layers.26.mlp.up_proj` | 7.6952 | 7.8803 |
| `model.layers.26.mlp.down_proj` | 7.6952 | 8.6390 |
| `model.layers.27.self_attn.q_proj` | 7.6952 | 7.7038 |
| `model.layers.27.self_attn.k_proj` | 7.6952 | 7.7012 |
| `model.layers.27.self_attn.v_proj` | 7.6952 | 7.7618 |
| `model.layers.27.self_attn.o_proj` | 7.6952 | 7.7132 |
| `model.layers.27.mlp.gate_proj` | 7.6952 | 7.9829 |
| `model.layers.27.mlp.up_proj` | 7.6952 | 7.9034 |
| `model.layers.27.mlp.down_proj` | 7.6952 | 8.6901 |
| `model.layers.28.self_attn.q_proj` | 7.6952 | 7.7177 |
| `model.layers.28.self_attn.k_proj` | 7.6952 | 7.6924 |
| `model.layers.28.self_attn.v_proj` | 7.6952 | 7.7682 |
| `model.layers.28.self_attn.o_proj` | 7.6952 | 7.7277 |
| `model.layers.28.mlp.gate_proj` | 7.6952 | 7.9870 |
| `model.layers.28.mlp.up_proj` | 7.6952 | 7.8986 |
| `model.layers.28.mlp.down_proj` | 7.6952 | 8.7012 |
| `model.layers.29.self_attn.q_proj` | 7.6952 | 7.7055 |
| `model.layers.29.self_attn.k_proj` | 7.6952 | 7.7034 |
| `model.layers.29.self_attn.v_proj` | 7.6952 | 7.7825 |
| `model.layers.29.self_attn.o_proj` | 7.6952 | 7.7252 |
| `model.layers.29.mlp.gate_proj` | 7.6952 | 8.0840 |
| `model.layers.29.mlp.up_proj` | 7.6952 | 7.8915 |
| `model.layers.29.mlp.down_proj` | 7.6952 | 8.7479 |
| `model.layers.30.self_attn.q_proj` | 7.6952 | 7.7234 |
| `model.layers.30.self_attn.k_proj` | 7.6952 | 7.7286 |
| `model.layers.30.self_attn.v_proj` | 7.6952 | 7.7963 |
| `model.layers.30.self_attn.o_proj` | 7.6952 | 7.7490 |
| `model.layers.30.mlp.gate_proj` | 7.6952 | 8.0232 |
| `model.layers.30.mlp.up_proj` | 7.6952 | 7.9792 |
| `model.layers.30.mlp.down_proj` | 7.6952 | 8.8506 |
| `model.layers.31.self_attn.q_proj` | 7.6952 | 7.7214 |
| `model.layers.31.self_attn.k_proj` | 7.6952 | 7.7139 |
| `model.layers.31.self_attn.v_proj` | 7.6952 | 7.8211 |
| `model.layers.31.self_attn.o_proj` | 7.6952 | 7.7373 |
| `model.layers.31.mlp.gate_proj` | 7.6952 | 8.1832 |
| `model.layers.31.mlp.up_proj` | 7.6952 | 8.2042 |
| `model.layers.31.mlp.down_proj` | 7.6952 | 10.0703 |

## Loss Delta Grid

| Layer | baseline | 8192 |
| --- | --- | --- |
| `model.layers.0.self_attn.q_proj` | 0.0000 | 0.0002 |
| `model.layers.0.self_attn.k_proj` | 0.0000 | -0.0039 |
| `model.layers.0.self_attn.v_proj` | 0.0000 | 3.8921 |
| `model.layers.0.self_attn.o_proj` | 0.0000 | 0.0361 |
| `model.layers.0.mlp.gate_proj` | 0.0000 | -0.0012 |
| `model.layers.0.mlp.up_proj` | 0.0000 | -0.0025 |
| `model.layers.0.mlp.down_proj` | 0.0000 | 0.7358 |
| `model.layers.1.self_attn.q_proj` | 0.0000 | 0.0071 |
| `model.layers.1.self_attn.k_proj` | 0.0000 | 0.0080 |
| `model.layers.1.self_attn.v_proj` | 0.0000 | 0.7826 |
| `model.layers.1.self_attn.o_proj` | 0.0000 | 0.0230 |
| `model.layers.1.mlp.gate_proj` | 0.0000 | 0.0061 |
| `model.layers.1.mlp.up_proj` | 0.0000 | 0.0113 |
| `model.layers.1.mlp.down_proj` | 0.0000 | 5.1255 |
| `model.layers.2.self_attn.q_proj` | 0.0000 | -0.0000 |
| `model.layers.2.self_attn.k_proj` | 0.0000 | 0.0046 |
| `model.layers.2.self_attn.v_proj` | 0.0000 | 0.0549 |
| `model.layers.2.self_attn.o_proj` | 0.0000 | 0.0101 |
| `model.layers.2.mlp.gate_proj` | 0.0000 | 0.0139 |
| `model.layers.2.mlp.up_proj` | 0.0000 | 0.0090 |
| `model.layers.2.mlp.down_proj` | 0.0000 | 0.1489 |
| `model.layers.3.self_attn.q_proj` | 0.0000 | 0.0030 |
| `model.layers.3.self_attn.k_proj` | 0.0000 | 0.0012 |
| `model.layers.3.self_attn.v_proj` | 0.0000 | 0.0911 |
| `model.layers.3.self_attn.o_proj` | 0.0000 | 0.0036 |
| `model.layers.3.mlp.gate_proj` | 0.0000 | 0.0118 |
| `model.layers.3.mlp.up_proj` | 0.0000 | 0.0143 |
| `model.layers.3.mlp.down_proj` | 0.0000 | 0.1232 |
| `model.layers.4.self_attn.q_proj` | 0.0000 | 0.0023 |
| `model.layers.4.self_attn.k_proj` | 0.0000 | 0.0037 |
| `model.layers.4.self_attn.v_proj` | 0.0000 | 0.0506 |
| `model.layers.4.self_attn.o_proj` | 0.0000 | 0.0079 |
| `model.layers.4.mlp.gate_proj` | 0.0000 | 0.0168 |
| `model.layers.4.mlp.up_proj` | 0.0000 | 0.0207 |
| `model.layers.4.mlp.down_proj` | 0.0000 | 0.1249 |
| `model.layers.5.self_attn.q_proj` | 0.0000 | 0.0016 |
| `model.layers.5.self_attn.k_proj` | 0.0000 | 0.0063 |
| `model.layers.5.self_attn.v_proj` | 0.0000 | 0.0376 |
| `model.layers.5.self_attn.o_proj` | 0.0000 | 0.0072 |
| `model.layers.5.mlp.gate_proj` | 0.0000 | 0.0047 |
| `model.layers.5.mlp.up_proj` | 0.0000 | 0.0191 |
| `model.layers.5.mlp.down_proj` | 0.0000 | 0.1406 |
| `model.layers.6.self_attn.q_proj` | 0.0000 | 0.0053 |
| `model.layers.6.self_attn.k_proj` | 0.0000 | 0.0038 |
| `model.layers.6.self_attn.v_proj` | 0.0000 | 0.0238 |
| `model.layers.6.self_attn.o_proj` | 0.0000 | 0.0071 |
| `model.layers.6.mlp.gate_proj` | 0.0000 | 0.0049 |
| `model.layers.6.mlp.up_proj` | 0.0000 | 0.0185 |
| `model.layers.6.mlp.down_proj` | 0.0000 | 0.1333 |
| `model.layers.7.self_attn.q_proj` | 0.0000 | 0.0036 |
| `model.layers.7.self_attn.k_proj` | 0.0000 | 0.0008 |
| `model.layers.7.self_attn.v_proj` | 0.0000 | 0.0195 |
| `model.layers.7.self_attn.o_proj` | 0.0000 | 0.0123 |
| `model.layers.7.mlp.gate_proj` | 0.0000 | 0.0097 |
| `model.layers.7.mlp.up_proj` | 0.0000 | 0.0149 |
| `model.layers.7.mlp.down_proj` | 0.0000 | 0.1279 |
| `model.layers.8.self_attn.q_proj` | 0.0000 | 0.0015 |
| `model.layers.8.self_attn.k_proj` | 0.0000 | 0.0016 |
| `model.layers.8.self_attn.v_proj` | 0.0000 | 0.0158 |
| `model.layers.8.self_attn.o_proj` | 0.0000 | 0.0089 |
| `model.layers.8.mlp.gate_proj` | 0.0000 | 0.0065 |
| `model.layers.8.mlp.up_proj` | 0.0000 | 0.0108 |
| `model.layers.8.mlp.down_proj` | 0.0000 | 0.1019 |
| `model.layers.9.self_attn.q_proj` | 0.0000 | 0.0061 |
| `model.layers.9.self_attn.k_proj` | 0.0000 | 0.0022 |
| `model.layers.9.self_attn.v_proj` | 0.0000 | 0.0172 |
| `model.layers.9.self_attn.o_proj` | 0.0000 | 0.0035 |
| `model.layers.9.mlp.gate_proj` | 0.0000 | 0.0093 |
| `model.layers.9.mlp.up_proj` | 0.0000 | 0.0145 |
| `model.layers.9.mlp.down_proj` | 0.0000 | 0.0909 |
| `model.layers.10.self_attn.q_proj` | 0.0000 | 0.0039 |
| `model.layers.10.self_attn.k_proj` | 0.0000 | 0.0022 |
| `model.layers.10.self_attn.v_proj` | 0.0000 | 0.0172 |
| `model.layers.10.self_attn.o_proj` | 0.0000 | 0.0115 |
| `model.layers.10.mlp.gate_proj` | 0.0000 | 0.0110 |
| `model.layers.10.mlp.up_proj` | 0.0000 | 0.0186 |
| `model.layers.10.mlp.down_proj` | 0.0000 | 0.0791 |
| `model.layers.11.self_attn.q_proj` | 0.0000 | 0.0010 |
| `model.layers.11.self_attn.k_proj` | 0.0000 | 0.0013 |
| `model.layers.11.self_attn.v_proj` | 0.0000 | 0.0197 |
| `model.layers.11.self_attn.o_proj` | 0.0000 | 0.0152 |
| `model.layers.11.mlp.gate_proj` | 0.0000 | 0.0091 |
| `model.layers.11.mlp.up_proj` | 0.0000 | 0.0145 |
| `model.layers.11.mlp.down_proj` | 0.0000 | 0.0781 |
| `model.layers.12.self_attn.q_proj` | 0.0000 | 0.0071 |
| `model.layers.12.self_attn.k_proj` | 0.0000 | 0.0034 |
| `model.layers.12.self_attn.v_proj` | 0.0000 | 0.0309 |
| `model.layers.12.self_attn.o_proj` | 0.0000 | 0.0153 |
| `model.layers.12.mlp.gate_proj` | 0.0000 | 0.0100 |
| `model.layers.12.mlp.up_proj` | 0.0000 | 0.0156 |
| `model.layers.12.mlp.down_proj` | 0.0000 | 0.0791 |
| `model.layers.13.self_attn.q_proj` | 0.0000 | 0.0053 |
| `model.layers.13.self_attn.k_proj` | 0.0000 | 0.0004 |
| `model.layers.13.self_attn.v_proj` | 0.0000 | 0.0153 |
| `model.layers.13.self_attn.o_proj` | 0.0000 | 0.0143 |
| `model.layers.13.mlp.gate_proj` | 0.0000 | 0.0096 |
| `model.layers.13.mlp.up_proj` | 0.0000 | 0.0093 |
| `model.layers.13.mlp.down_proj` | 0.0000 | 0.0631 |
| `model.layers.14.self_attn.q_proj` | 0.0000 | 0.0045 |
| `model.layers.14.self_attn.k_proj` | 0.0000 | 0.0010 |
| `model.layers.14.self_attn.v_proj` | 0.0000 | 0.0304 |
| `model.layers.14.self_attn.o_proj` | 0.0000 | 0.0124 |
| `model.layers.14.mlp.gate_proj` | 0.0000 | 0.0096 |
| `model.layers.14.mlp.up_proj` | 0.0000 | 0.0067 |
| `model.layers.14.mlp.down_proj` | 0.0000 | 0.0683 |
| `model.layers.15.self_attn.q_proj` | 0.0000 | 0.0021 |
| `model.layers.15.self_attn.k_proj` | 0.0000 | 0.0037 |
| `model.layers.15.self_attn.v_proj` | 0.0000 | 0.0289 |
| `model.layers.15.self_attn.o_proj` | 0.0000 | 0.0186 |
| `model.layers.15.mlp.gate_proj` | 0.0000 | 0.0091 |
| `model.layers.15.mlp.up_proj` | 0.0000 | 0.0093 |
| `model.layers.15.mlp.down_proj` | 0.0000 | 0.0753 |
| `model.layers.16.self_attn.q_proj` | 0.0000 | 0.0045 |
| `model.layers.16.self_attn.k_proj` | 0.0000 | 0.0030 |
| `model.layers.16.self_attn.v_proj` | 0.0000 | 0.0118 |
| `model.layers.16.self_attn.o_proj` | 0.0000 | 0.0135 |
| `model.layers.16.mlp.gate_proj` | 0.0000 | 0.0103 |
| `model.layers.16.mlp.up_proj` | 0.0000 | 0.0191 |
| `model.layers.16.mlp.down_proj` | 0.0000 | 0.0983 |
| `model.layers.17.self_attn.q_proj` | 0.0000 | 0.0030 |
| `model.layers.17.self_attn.k_proj` | 0.0000 | 0.0024 |
| `model.layers.17.self_attn.v_proj` | 0.0000 | 0.0238 |
| `model.layers.17.self_attn.o_proj` | 0.0000 | 0.0056 |
| `model.layers.17.mlp.gate_proj` | 0.0000 | 0.0120 |
| `model.layers.17.mlp.up_proj` | 0.0000 | 0.0199 |
| `model.layers.17.mlp.down_proj` | 0.0000 | 0.1224 |
| `model.layers.18.self_attn.q_proj` | 0.0000 | 0.0003 |
| `model.layers.18.self_attn.k_proj` | 0.0000 | 0.0037 |
| `model.layers.18.self_attn.v_proj` | 0.0000 | 0.0168 |
| `model.layers.18.self_attn.o_proj` | 0.0000 | 0.0136 |
| `model.layers.18.mlp.gate_proj` | 0.0000 | 0.0181 |
| `model.layers.18.mlp.up_proj` | 0.0000 | 0.0291 |
| `model.layers.18.mlp.down_proj` | 0.0000 | 0.1922 |
| `model.layers.19.self_attn.q_proj` | 0.0000 | 0.0065 |
| `model.layers.19.self_attn.k_proj` | 0.0000 | 0.0050 |
| `model.layers.19.self_attn.v_proj` | 0.0000 | 0.0238 |
| `model.layers.19.self_attn.o_proj` | 0.0000 | 0.0113 |
| `model.layers.19.mlp.gate_proj` | 0.0000 | 0.0217 |
| `model.layers.19.mlp.up_proj` | 0.0000 | 0.0183 |
| `model.layers.19.mlp.down_proj` | 0.0000 | 0.2299 |
| `model.layers.20.self_attn.q_proj` | 0.0000 | 0.0040 |
| `model.layers.20.self_attn.k_proj` | 0.0000 | 0.0017 |
| `model.layers.20.self_attn.v_proj` | 0.0000 | 0.0165 |
| `model.layers.20.self_attn.o_proj` | 0.0000 | 0.0129 |
| `model.layers.20.mlp.gate_proj` | 0.0000 | 0.0238 |
| `model.layers.20.mlp.up_proj` | 0.0000 | 0.0290 |
| `model.layers.20.mlp.down_proj` | 0.0000 | 0.1982 |
| `model.layers.21.self_attn.q_proj` | 0.0000 | 0.0018 |
| `model.layers.21.self_attn.k_proj` | 0.0000 | 0.0016 |
| `model.layers.21.self_attn.v_proj` | 0.0000 | 0.0106 |
| `model.layers.21.self_attn.o_proj` | 0.0000 | 0.0061 |
| `model.layers.21.mlp.gate_proj` | 0.0000 | 0.0240 |
| `model.layers.21.mlp.up_proj` | 0.0000 | 0.0266 |
| `model.layers.21.mlp.down_proj` | 0.0000 | 0.1789 |
| `model.layers.22.self_attn.q_proj` | 0.0000 | 0.0033 |
| `model.layers.22.self_attn.k_proj` | 0.0000 | 0.0010 |
| `model.layers.22.self_attn.v_proj` | 0.0000 | 0.0092 |
| `model.layers.22.self_attn.o_proj` | 0.0000 | 0.0049 |
| `model.layers.22.mlp.gate_proj` | 0.0000 | 0.0213 |
| `model.layers.22.mlp.up_proj` | 0.0000 | 0.0242 |
| `model.layers.22.mlp.down_proj` | 0.0000 | 0.1581 |
| `model.layers.23.self_attn.q_proj` | 0.0000 | 0.0032 |
| `model.layers.23.self_attn.k_proj` | 0.0000 | 0.0006 |
| `model.layers.23.self_attn.v_proj` | 0.0000 | 0.0127 |
| `model.layers.23.self_attn.o_proj` | 0.0000 | 0.0034 |
| `model.layers.23.mlp.gate_proj` | 0.0000 | 0.0276 |
| `model.layers.23.mlp.up_proj` | 0.0000 | 0.0197 |
| `model.layers.23.mlp.down_proj` | 0.0000 | 0.1410 |
| `model.layers.24.self_attn.q_proj` | 0.0000 | 0.0029 |
| `model.layers.24.self_attn.k_proj` | 0.0000 | 0.0005 |
| `model.layers.24.self_attn.v_proj` | 0.0000 | 0.0089 |
| `model.layers.24.self_attn.o_proj` | 0.0000 | 0.0043 |
| `model.layers.24.mlp.gate_proj` | 0.0000 | 0.0302 |
| `model.layers.24.mlp.up_proj` | 0.0000 | 0.0231 |
| `model.layers.24.mlp.down_proj` | 0.0000 | 0.1333 |
| `model.layers.25.self_attn.q_proj` | 0.0000 | 0.0018 |
| `model.layers.25.self_attn.k_proj` | 0.0000 | 0.0021 |
| `model.layers.25.self_attn.v_proj` | 0.0000 | 0.0082 |
| `model.layers.25.self_attn.o_proj` | 0.0000 | 0.0040 |
| `model.layers.25.mlp.gate_proj` | 0.0000 | 0.0397 |
| `model.layers.25.mlp.up_proj` | 0.0000 | 0.0302 |
| `model.layers.25.mlp.down_proj` | 0.0000 | 0.1292 |
| `model.layers.26.self_attn.q_proj` | 0.0000 | 0.0045 |
| `model.layers.26.self_attn.k_proj` | 0.0000 | 0.0010 |
| `model.layers.26.self_attn.v_proj` | 0.0000 | 0.0102 |
| `model.layers.26.self_attn.o_proj` | 0.0000 | 0.0027 |
| `model.layers.26.mlp.gate_proj` | 0.0000 | 0.0369 |
| `model.layers.26.mlp.up_proj` | 0.0000 | 0.0238 |
| `model.layers.26.mlp.down_proj` | 0.0000 | 0.1157 |
| `model.layers.27.self_attn.q_proj` | 0.0000 | 0.0011 |
| `model.layers.27.self_attn.k_proj` | 0.0000 | 0.0008 |
| `model.layers.27.self_attn.v_proj` | 0.0000 | 0.0086 |
| `model.layers.27.self_attn.o_proj` | 0.0000 | 0.0023 |
| `model.layers.27.mlp.gate_proj` | 0.0000 | 0.0367 |
| `model.layers.27.mlp.up_proj` | 0.0000 | 0.0267 |
| `model.layers.27.mlp.down_proj` | 0.0000 | 0.1216 |
| `model.layers.28.self_attn.q_proj` | 0.0000 | 0.0029 |
| `model.layers.28.self_attn.k_proj` | 0.0000 | -0.0004 |
| `model.layers.28.self_attn.v_proj` | 0.0000 | 0.0094 |
| `model.layers.28.self_attn.o_proj` | 0.0000 | 0.0042 |
| `model.layers.28.mlp.gate_proj` | 0.0000 | 0.0372 |
| `model.layers.28.mlp.up_proj` | 0.0000 | 0.0261 |
| `model.layers.28.mlp.down_proj` | 0.0000 | 0.1229 |
| `model.layers.29.self_attn.q_proj` | 0.0000 | 0.0013 |
| `model.layers.29.self_attn.k_proj` | 0.0000 | 0.0011 |
| `model.layers.29.self_attn.v_proj` | 0.0000 | 0.0113 |
| `model.layers.29.self_attn.o_proj` | 0.0000 | 0.0039 |
| `model.layers.29.mlp.gate_proj` | 0.0000 | 0.0493 |
| `model.layers.29.mlp.up_proj` | 0.0000 | 0.0252 |
| `model.layers.29.mlp.down_proj` | 0.0000 | 0.1282 |
| `model.layers.30.self_attn.q_proj` | 0.0000 | 0.0037 |
| `model.layers.30.self_attn.k_proj` | 0.0000 | 0.0043 |
| `model.layers.30.self_attn.v_proj` | 0.0000 | 0.0131 |
| `model.layers.30.self_attn.o_proj` | 0.0000 | 0.0070 |
| `model.layers.30.mlp.gate_proj` | 0.0000 | 0.0417 |
| `model.layers.30.mlp.up_proj` | 0.0000 | 0.0362 |
| `model.layers.30.mlp.down_proj` | 0.0000 | 0.1399 |
| `model.layers.31.self_attn.q_proj` | 0.0000 | 0.0034 |
| `model.layers.31.self_attn.k_proj` | 0.0000 | 0.0024 |
| `model.layers.31.self_attn.v_proj` | 0.0000 | 0.0162 |
| `model.layers.31.self_attn.o_proj` | 0.0000 | 0.0055 |
| `model.layers.31.mlp.gate_proj` | 0.0000 | 0.0615 |
| `model.layers.31.mlp.up_proj` | 0.0000 | 0.0640 |
| `model.layers.31.mlp.down_proj` | 0.0000 | 0.2690 |

## Layer Metadata

| Layer | Weight shape | Best k | Best ppl | Worst k | Worst ppl |
| --- | --- | ---: | ---: | ---: | ---: |
| `model.layers.0.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.6966 | 8192 | 7.6966 |
| `model.layers.0.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.6654 | 8192 | 7.6654 |
| `model.layers.0.self_attn.v_proj` | `(1024, 4096)` | 8192 | 377.1837 | 8192 | 377.1837 |
| `model.layers.0.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.9777 | 8192 | 7.9777 |
| `model.layers.0.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.6861 | 8192 | 7.6861 |
| `model.layers.0.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.6761 | 8192 | 7.6761 |
| `model.layers.0.mlp.down_proj` | `(4096, 14336)` | 8192 | 16.0618 | 8192 | 16.0618 |
| `model.layers.1.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7499 | 8192 | 7.7499 |
| `model.layers.1.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7571 | 8192 | 7.7571 |
| `model.layers.1.self_attn.v_proj` | `(1024, 4096)` | 8192 | 16.8308 | 8192 | 16.8308 |
| `model.layers.1.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.8739 | 8192 | 7.8739 |
| `model.layers.1.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7425 | 8192 | 7.7425 |
| `model.layers.1.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.7825 | 8192 | 7.7825 |
| `model.layers.1.mlp.down_proj` | `(4096, 14336)` | 8192 | 1294.8345 | 8192 | 1294.8345 |
| `model.layers.2.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.6950 | 8192 | 7.6950 |
| `model.layers.2.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7305 | 8192 | 7.7305 |
| `model.layers.2.self_attn.v_proj` | `(1024, 4096)` | 8192 | 8.1298 | 8192 | 8.1298 |
| `model.layers.2.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7736 | 8192 | 7.7736 |
| `model.layers.2.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.8029 | 8192 | 7.8029 |
| `model.layers.2.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.7647 | 8192 | 7.7647 |
| `model.layers.2.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.9304 | 8192 | 8.9304 |
| `model.layers.3.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7184 | 8192 | 7.7184 |
| `model.layers.3.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7047 | 8192 | 7.7047 |
| `model.layers.3.self_attn.v_proj` | `(1024, 4096)` | 8192 | 8.4296 | 8192 | 8.4296 |
| `model.layers.3.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7233 | 8192 | 7.7233 |
| `model.layers.3.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7865 | 8192 | 7.7865 |
| `model.layers.3.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8063 | 8192 | 7.8063 |
| `model.layers.3.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.7045 | 8192 | 8.7045 |
| `model.layers.4.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7130 | 8192 | 7.7130 |
| `model.layers.4.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7237 | 8192 | 7.7237 |
| `model.layers.4.self_attn.v_proj` | `(1024, 4096)` | 8192 | 8.0945 | 8192 | 8.0945 |
| `model.layers.4.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7565 | 8192 | 7.7565 |
| `model.layers.4.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.8256 | 8192 | 7.8256 |
| `model.layers.4.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8560 | 8192 | 7.8560 |
| `model.layers.4.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.7192 | 8192 | 8.7192 |
| `model.layers.5.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7072 | 8192 | 7.7072 |
| `model.layers.5.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7441 | 8192 | 7.7441 |
| `model.layers.5.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.9900 | 8192 | 7.9900 |
| `model.layers.5.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7507 | 8192 | 7.7507 |
| `model.layers.5.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7312 | 8192 | 7.7312 |
| `model.layers.5.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8439 | 8192 | 7.8439 |
| `model.layers.5.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.8567 | 8192 | 8.8567 |
| `model.layers.6.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7363 | 8192 | 7.7363 |
| `model.layers.6.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7248 | 8192 | 7.7248 |
| `model.layers.6.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8808 | 8192 | 7.8808 |
| `model.layers.6.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7501 | 8192 | 7.7501 |
| `model.layers.6.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7331 | 8192 | 7.7331 |
| `model.layers.6.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8389 | 8192 | 7.8389 |
| `model.layers.6.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.7921 | 8192 | 8.7921 |
| `model.layers.7.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7229 | 8192 | 7.7229 |
| `model.layers.7.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7014 | 8192 | 7.7014 |
| `model.layers.7.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8467 | 8192 | 7.8467 |
| `model.layers.7.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7903 | 8192 | 7.7903 |
| `model.layers.7.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7702 | 8192 | 7.7702 |
| `model.layers.7.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8107 | 8192 | 7.8107 |
| `model.layers.7.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.7455 | 8192 | 8.7455 |
| `model.layers.8.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7066 | 8192 | 7.7066 |
| `model.layers.8.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7076 | 8192 | 7.7076 |
| `model.layers.8.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8178 | 8192 | 7.8178 |
| `model.layers.8.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7639 | 8192 | 7.7639 |
| `model.layers.8.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7452 | 8192 | 7.7452 |
| `model.layers.8.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.7786 | 8192 | 7.7786 |
| `model.layers.8.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.5205 | 8192 | 8.5205 |
| `model.layers.9.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7423 | 8192 | 7.7423 |
| `model.layers.9.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7119 | 8192 | 7.7119 |
| `model.layers.9.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8289 | 8192 | 7.8289 |
| `model.layers.9.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7222 | 8192 | 7.7222 |
| `model.layers.9.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7671 | 8192 | 7.7671 |
| `model.layers.9.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8076 | 8192 | 7.8076 |
| `model.layers.9.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.4275 | 8192 | 8.4275 |
| `model.layers.10.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7252 | 8192 | 7.7252 |
| `model.layers.10.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7122 | 8192 | 7.7122 |
| `model.layers.10.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8284 | 8192 | 7.8284 |
| `model.layers.10.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7842 | 8192 | 7.7842 |
| `model.layers.10.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7800 | 8192 | 7.7800 |
| `model.layers.10.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8393 | 8192 | 7.8393 |
| `model.layers.10.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.3285 | 8192 | 8.3285 |
| `model.layers.11.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7031 | 8192 | 7.7031 |
| `model.layers.11.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7053 | 8192 | 7.7053 |
| `model.layers.11.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8481 | 8192 | 7.8481 |
| `model.layers.11.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.8131 | 8192 | 7.8131 |
| `model.layers.11.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7654 | 8192 | 7.7654 |
| `model.layers.11.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8080 | 8192 | 7.8080 |
| `model.layers.11.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.3204 | 8192 | 8.3204 |
| `model.layers.12.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7504 | 8192 | 7.7504 |
| `model.layers.12.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7216 | 8192 | 7.7216 |
| `model.layers.12.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.9365 | 8192 | 7.9365 |
| `model.layers.12.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.8142 | 8192 | 7.8142 |
| `model.layers.12.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7723 | 8192 | 7.7723 |
| `model.layers.12.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8165 | 8192 | 7.8165 |
| `model.layers.12.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.3285 | 8192 | 8.3285 |
| `model.layers.13.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7364 | 8192 | 7.7364 |
| `model.layers.13.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.6985 | 8192 | 7.6985 |
| `model.layers.13.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8139 | 8192 | 7.8139 |
| `model.layers.13.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.8060 | 8192 | 7.8060 |
| `model.layers.13.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7694 | 8192 | 7.7694 |
| `model.layers.13.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.7671 | 8192 | 7.7671 |
| `model.layers.13.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.1962 | 8192 | 8.1962 |
| `model.layers.14.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7301 | 8192 | 7.7301 |
| `model.layers.14.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7028 | 8192 | 7.7028 |
| `model.layers.14.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.9326 | 8192 | 7.9326 |
| `model.layers.14.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7909 | 8192 | 7.7909 |
| `model.layers.14.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7697 | 8192 | 7.7697 |
| `model.layers.14.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.7472 | 8192 | 7.7472 |
| `model.layers.14.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.2392 | 8192 | 8.2392 |
| `model.layers.15.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7114 | 8192 | 7.7114 |
| `model.layers.15.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7241 | 8192 | 7.7241 |
| `model.layers.15.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.9205 | 8192 | 7.9205 |
| `model.layers.15.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.8400 | 8192 | 7.8400 |
| `model.layers.15.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7659 | 8192 | 7.7659 |
| `model.layers.15.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.7669 | 8192 | 7.7669 |
| `model.layers.15.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.2970 | 8192 | 8.2970 |
| `model.layers.16.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7301 | 8192 | 7.7301 |
| `model.layers.16.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7184 | 8192 | 7.7184 |
| `model.layers.16.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7866 | 8192 | 7.7866 |
| `model.layers.16.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.8000 | 8192 | 7.8000 |
| `model.layers.16.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7746 | 8192 | 7.7746 |
| `model.layers.16.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8435 | 8192 | 7.8435 |
| `model.layers.16.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.4903 | 8192 | 8.4903 |
| `model.layers.17.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7181 | 8192 | 7.7181 |
| `model.layers.17.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7139 | 8192 | 7.7139 |
| `model.layers.17.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8803 | 8192 | 7.8803 |
| `model.layers.17.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7387 | 8192 | 7.7387 |
| `model.layers.17.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.7884 | 8192 | 7.7884 |
| `model.layers.17.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8499 | 8192 | 7.8499 |
| `model.layers.17.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.6974 | 8192 | 8.6974 |
| `model.layers.18.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.6973 | 8192 | 7.6973 |
| `model.layers.18.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7241 | 8192 | 7.7241 |
| `model.layers.18.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8256 | 8192 | 7.8256 |
| `model.layers.18.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.8003 | 8192 | 7.8003 |
| `model.layers.18.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.8361 | 8192 | 7.8361 |
| `model.layers.18.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.9223 | 8192 | 7.9223 |
| `model.layers.18.mlp.down_proj` | `(4096, 14336)` | 8192 | 9.3264 | 8192 | 9.3264 |
| `model.layers.19.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7454 | 8192 | 7.7454 |
| `model.layers.19.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7341 | 8192 | 7.7341 |
| `model.layers.19.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8805 | 8192 | 7.8805 |
| `model.layers.19.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7830 | 8192 | 7.7830 |
| `model.layers.19.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.8638 | 8192 | 7.8638 |
| `model.layers.19.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8377 | 8192 | 7.8377 |
| `model.layers.19.mlp.down_proj` | `(4096, 14336)` | 8192 | 9.6844 | 8192 | 9.6844 |
| `model.layers.20.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7263 | 8192 | 7.7263 |
| `model.layers.20.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7084 | 8192 | 7.7084 |
| `model.layers.20.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8232 | 8192 | 7.8232 |
| `model.layers.20.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7951 | 8192 | 7.7951 |
| `model.layers.20.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.8802 | 8192 | 7.8802 |
| `model.layers.20.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.9218 | 8192 | 7.9218 |
| `model.layers.20.mlp.down_proj` | `(4096, 14336)` | 8192 | 9.3816 | 8192 | 9.3816 |
| `model.layers.21.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7091 | 8192 | 7.7091 |
| `model.layers.21.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7076 | 8192 | 7.7076 |
| `model.layers.21.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7771 | 8192 | 7.7771 |
| `model.layers.21.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7421 | 8192 | 7.7421 |
| `model.layers.21.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.8819 | 8192 | 7.8819 |
| `model.layers.21.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.9025 | 8192 | 7.9025 |
| `model.layers.21.mlp.down_proj` | `(4096, 14336)` | 8192 | 9.2029 | 8192 | 9.2029 |
| `model.layers.22.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7206 | 8192 | 7.7206 |
| `model.layers.22.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7026 | 8192 | 7.7026 |
| `model.layers.22.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7663 | 8192 | 7.7663 |
| `model.layers.22.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7334 | 8192 | 7.7334 |
| `model.layers.22.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.8607 | 8192 | 7.8607 |
| `model.layers.22.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8833 | 8192 | 7.8833 |
| `model.layers.22.mlp.down_proj` | `(4096, 14336)` | 8192 | 9.0134 | 8192 | 9.0134 |
| `model.layers.23.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7198 | 8192 | 7.7198 |
| `model.layers.23.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.6997 | 8192 | 7.6997 |
| `model.layers.23.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7939 | 8192 | 7.7939 |
| `model.layers.23.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7214 | 8192 | 7.7214 |
| `model.layers.23.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.9102 | 8192 | 7.9102 |
| `model.layers.23.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8482 | 8192 | 7.8482 |
| `model.layers.23.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.8605 | 8192 | 8.8605 |
| `model.layers.24.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7178 | 8192 | 7.7178 |
| `model.layers.24.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.6988 | 8192 | 7.6988 |
| `model.layers.24.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7641 | 8192 | 7.7641 |
| `model.layers.24.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7282 | 8192 | 7.7282 |
| `model.layers.24.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.9310 | 8192 | 7.9310 |
| `model.layers.24.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8753 | 8192 | 7.8753 |
| `model.layers.24.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.7923 | 8192 | 8.7923 |
| `model.layers.25.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7090 | 8192 | 7.7090 |
| `model.layers.25.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7114 | 8192 | 7.7114 |
| `model.layers.25.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7587 | 8192 | 7.7587 |
| `model.layers.25.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7261 | 8192 | 7.7261 |
| `model.layers.25.mlp.gate_proj` | `(14336, 4096)` | 8192 | 8.0066 | 8192 | 8.0066 |
| `model.layers.25.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.9310 | 8192 | 7.9310 |
| `model.layers.25.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.7561 | 8192 | 8.7561 |
| `model.layers.26.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7299 | 8192 | 7.7299 |
| `model.layers.26.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7032 | 8192 | 7.7032 |
| `model.layers.26.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7740 | 8192 | 7.7740 |
| `model.layers.26.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7161 | 8192 | 7.7161 |
| `model.layers.26.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.9847 | 8192 | 7.9847 |
| `model.layers.26.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8803 | 8192 | 7.8803 |
| `model.layers.26.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.6390 | 8192 | 8.6390 |
| `model.layers.27.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7038 | 8192 | 7.7038 |
| `model.layers.27.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7012 | 8192 | 7.7012 |
| `model.layers.27.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7618 | 8192 | 7.7618 |
| `model.layers.27.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7132 | 8192 | 7.7132 |
| `model.layers.27.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.9829 | 8192 | 7.9829 |
| `model.layers.27.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.9034 | 8192 | 7.9034 |
| `model.layers.27.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.6901 | 8192 | 8.6901 |
| `model.layers.28.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7177 | 8192 | 7.7177 |
| `model.layers.28.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.6924 | 8192 | 7.6924 |
| `model.layers.28.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7682 | 8192 | 7.7682 |
| `model.layers.28.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7277 | 8192 | 7.7277 |
| `model.layers.28.mlp.gate_proj` | `(14336, 4096)` | 8192 | 7.9870 | 8192 | 7.9870 |
| `model.layers.28.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8986 | 8192 | 7.8986 |
| `model.layers.28.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.7012 | 8192 | 8.7012 |
| `model.layers.29.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7055 | 8192 | 7.7055 |
| `model.layers.29.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7034 | 8192 | 7.7034 |
| `model.layers.29.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7825 | 8192 | 7.7825 |
| `model.layers.29.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7252 | 8192 | 7.7252 |
| `model.layers.29.mlp.gate_proj` | `(14336, 4096)` | 8192 | 8.0840 | 8192 | 8.0840 |
| `model.layers.29.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.8915 | 8192 | 7.8915 |
| `model.layers.29.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.7479 | 8192 | 8.7479 |
| `model.layers.30.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7234 | 8192 | 7.7234 |
| `model.layers.30.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7286 | 8192 | 7.7286 |
| `model.layers.30.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.7963 | 8192 | 7.7963 |
| `model.layers.30.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7490 | 8192 | 7.7490 |
| `model.layers.30.mlp.gate_proj` | `(14336, 4096)` | 8192 | 8.0232 | 8192 | 8.0232 |
| `model.layers.30.mlp.up_proj` | `(14336, 4096)` | 8192 | 7.9792 | 8192 | 7.9792 |
| `model.layers.30.mlp.down_proj` | `(4096, 14336)` | 8192 | 8.8506 | 8192 | 8.8506 |
| `model.layers.31.self_attn.q_proj` | `(4096, 4096)` | 8192 | 7.7214 | 8192 | 7.7214 |
| `model.layers.31.self_attn.k_proj` | `(1024, 4096)` | 8192 | 7.7139 | 8192 | 7.7139 |
| `model.layers.31.self_attn.v_proj` | `(1024, 4096)` | 8192 | 7.8211 | 8192 | 7.8211 |
| `model.layers.31.self_attn.o_proj` | `(4096, 4096)` | 8192 | 7.7373 | 8192 | 7.7373 |
| `model.layers.31.mlp.gate_proj` | `(14336, 4096)` | 8192 | 8.1832 | 8192 | 8.1832 |
| `model.layers.31.mlp.up_proj` | `(14336, 4096)` | 8192 | 8.2042 | 8192 | 8.2042 |
| `model.layers.31.mlp.down_proj` | `(4096, 14336)` | 8192 | 10.0703 | 8192 | 10.0703 |
