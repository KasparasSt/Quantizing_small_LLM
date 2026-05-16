# M3 Single-Layer Sensitivity Report

This report applies M3 to exactly one transformer `nn.Linear` layer at a time
and evaluates a grid of `k` values for that one layer.

## Settings

- `MODEL_PATH`: `checkpoints/mistral_7b_instruct_v03_sparsegpt_90`
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
| `model.layers.0.self_attn.q_proj` | 312.5392 | 324.1992 |
| `model.layers.0.self_attn.k_proj` | 312.5392 | 312.7727 |
| `model.layers.0.self_attn.v_proj` | 312.5392 | 83114.8267 |
| `model.layers.0.self_attn.o_proj` | 312.5392 | 362.9385 |
| `model.layers.0.mlp.gate_proj` | 312.5392 | 254.1718 |
| `model.layers.0.mlp.up_proj` | 312.5392 | 342.6199 |
| `model.layers.0.mlp.down_proj` | 312.5392 | 7310.7251 |
| `model.layers.1.self_attn.q_proj` | 312.5392 | 310.0183 |
| `model.layers.1.self_attn.k_proj` | 312.5392 | 325.3677 |
| `model.layers.1.self_attn.v_proj` | 312.5392 | 46241.1073 |
| `model.layers.1.self_attn.o_proj` | 312.5392 | 462.0041 |
| `model.layers.1.mlp.gate_proj` | 312.5392 | 278.8959 |
| `model.layers.1.mlp.up_proj` | 312.5392 | 338.9521 |
| `model.layers.1.mlp.down_proj` | 312.5392 | 77406.6199 |
| `model.layers.2.self_attn.q_proj` | 312.5392 | 317.0302 |
| `model.layers.2.self_attn.k_proj` | 312.5392 | 313.0373 |
| `model.layers.2.self_attn.v_proj` | 312.5392 | 3549.6047 |
| `model.layers.2.self_attn.o_proj` | 312.5392 | 446.1478 |
| `model.layers.2.mlp.gate_proj` | 312.5392 | 351.1852 |
| `model.layers.2.mlp.up_proj` | 312.5392 | 306.7409 |
| `model.layers.2.mlp.down_proj` | 312.5392 | 445.0670 |
| `model.layers.3.self_attn.q_proj` | 312.5392 | 314.5859 |
| `model.layers.3.self_attn.k_proj` | 312.5392 | 325.7234 |
| `model.layers.3.self_attn.v_proj` | 312.5392 | 3529.4682 |
| `model.layers.3.self_attn.o_proj` | 312.5392 | 386.2949 |
| `model.layers.3.mlp.gate_proj` | 312.5392 | 353.9159 |
| `model.layers.3.mlp.up_proj` | 312.5392 | 367.9400 |
| `model.layers.3.mlp.down_proj` | 312.5392 | 488.5048 |
| `model.layers.4.self_attn.q_proj` | 312.5392 | 315.6097 |
| `model.layers.4.self_attn.k_proj` | 312.5392 | 337.5957 |
| `model.layers.4.self_attn.v_proj` | 312.5392 | 57800.1001 |
| `model.layers.4.self_attn.o_proj` | 312.5392 | 395.5000 |
| `model.layers.4.mlp.gate_proj` | 312.5392 | 365.9487 |
| `model.layers.4.mlp.up_proj` | 312.5392 | 367.2004 |
| `model.layers.4.mlp.down_proj` | 312.5392 | 641.3038 |
| `model.layers.5.self_attn.q_proj` | 312.5392 | 311.0822 |
| `model.layers.5.self_attn.k_proj` | 312.5392 | 315.0707 |
| `model.layers.5.self_attn.v_proj` | 312.5392 | 774.0688 |
| `model.layers.5.self_attn.o_proj` | 312.5392 | 427.2294 |
| `model.layers.5.mlp.gate_proj` | 312.5392 | 339.2610 |
| `model.layers.5.mlp.up_proj` | 312.5392 | 336.2553 |
| `model.layers.5.mlp.down_proj` | 312.5392 | 622.1316 |
| `model.layers.6.self_attn.q_proj` | 312.5392 | 321.3569 |
| `model.layers.6.self_attn.k_proj` | 312.5392 | 312.2848 |
| `model.layers.6.self_attn.v_proj` | 312.5392 | 525.7204 |
| `model.layers.6.self_attn.o_proj` | 312.5392 | 346.4862 |
| `model.layers.6.mlp.gate_proj` | 312.5392 | 324.6501 |
| `model.layers.6.mlp.up_proj` | 312.5392 | 347.2855 |
| `model.layers.6.mlp.down_proj` | 312.5392 | 489.0584 |
| `model.layers.7.self_attn.q_proj` | 312.5392 | 314.8727 |
| `model.layers.7.self_attn.k_proj` | 312.5392 | 318.4802 |
| `model.layers.7.self_attn.v_proj` | 312.5392 | 491.0802 |
| `model.layers.7.self_attn.o_proj` | 312.5392 | 392.5139 |
| `model.layers.7.mlp.gate_proj` | 312.5392 | 332.9240 |
| `model.layers.7.mlp.up_proj` | 312.5392 | 323.1782 |
| `model.layers.7.mlp.down_proj` | 312.5392 | 395.1580 |
| `model.layers.8.self_attn.q_proj` | 312.5392 | 321.0604 |
| `model.layers.8.self_attn.k_proj` | 312.5392 | 317.6524 |
| `model.layers.8.self_attn.v_proj` | 312.5392 | 442.7207 |
| `model.layers.8.self_attn.o_proj` | 312.5392 | 344.9528 |
| `model.layers.8.mlp.gate_proj` | 312.5392 | 331.5877 |
| `model.layers.8.mlp.up_proj` | 312.5392 | 349.0940 |
| `model.layers.8.mlp.down_proj` | 312.5392 | 407.7899 |
| `model.layers.9.self_attn.q_proj` | 312.5392 | 317.6568 |
| `model.layers.9.self_attn.k_proj` | 312.5392 | 313.6879 |
| `model.layers.9.self_attn.v_proj` | 312.5392 | 427.8800 |
| `model.layers.9.self_attn.o_proj` | 312.5392 | 335.3298 |
| `model.layers.9.mlp.gate_proj` | 312.5392 | 343.3493 |
| `model.layers.9.mlp.up_proj` | 312.5392 | 341.0644 |
| `model.layers.9.mlp.down_proj` | 312.5392 | 457.8151 |
| `model.layers.10.self_attn.q_proj` | 312.5392 | 315.5471 |
| `model.layers.10.self_attn.k_proj` | 312.5392 | 317.1745 |
| `model.layers.10.self_attn.v_proj` | 312.5392 | 349.1036 |
| `model.layers.10.self_attn.o_proj` | 312.5392 | 379.1018 |
| `model.layers.10.mlp.gate_proj` | 312.5392 | 333.6819 |
| `model.layers.10.mlp.up_proj` | 312.5392 | 322.4343 |
| `model.layers.10.mlp.down_proj` | 312.5392 | 400.2514 |
| `model.layers.11.self_attn.q_proj` | 312.5392 | 319.0883 |
| `model.layers.11.self_attn.k_proj` | 312.5392 | 314.0070 |
| `model.layers.11.self_attn.v_proj` | 312.5392 | 421.7309 |
| `model.layers.11.self_attn.o_proj` | 312.5392 | 373.2205 |
| `model.layers.11.mlp.gate_proj` | 312.5392 | 325.2441 |
| `model.layers.11.mlp.up_proj` | 312.5392 | 317.2083 |
| `model.layers.11.mlp.down_proj` | 312.5392 | 365.2770 |
| `model.layers.12.self_attn.q_proj` | 312.5392 | 327.9992 |
| `model.layers.12.self_attn.k_proj` | 312.5392 | 320.9580 |
| `model.layers.12.self_attn.v_proj` | 312.5392 | 394.9543 |
| `model.layers.12.self_attn.o_proj` | 312.5392 | 354.3359 |
| `model.layers.12.mlp.gate_proj` | 312.5392 | 318.4440 |
| `model.layers.12.mlp.up_proj` | 312.5392 | 319.1065 |
| `model.layers.12.mlp.down_proj` | 312.5392 | 357.9694 |
| `model.layers.13.self_attn.q_proj` | 312.5392 | 317.6807 |
| `model.layers.13.self_attn.k_proj` | 312.5392 | 311.6374 |
| `model.layers.13.self_attn.v_proj` | 312.5392 | 375.3091 |
| `model.layers.13.self_attn.o_proj` | 312.5392 | 377.0205 |
| `model.layers.13.mlp.gate_proj` | 312.5392 | 326.4315 |
| `model.layers.13.mlp.up_proj` | 312.5392 | 341.6754 |
| `model.layers.13.mlp.down_proj` | 312.5392 | 368.6730 |
| `model.layers.14.self_attn.q_proj` | 312.5392 | 310.3702 |
| `model.layers.14.self_attn.k_proj` | 312.5392 | 316.9750 |
| `model.layers.14.self_attn.v_proj` | 312.5392 | 373.2399 |
| `model.layers.14.self_attn.o_proj` | 312.5392 | 342.4181 |
| `model.layers.14.mlp.gate_proj` | 312.5392 | 320.2517 |
| `model.layers.14.mlp.up_proj` | 312.5392 | 321.8136 |
| `model.layers.14.mlp.down_proj` | 312.5392 | 363.5491 |
| `model.layers.15.self_attn.q_proj` | 312.5392 | 314.1860 |
| `model.layers.15.self_attn.k_proj` | 312.5392 | 314.6753 |
| `model.layers.15.self_attn.v_proj` | 312.5392 | 400.0866 |
| `model.layers.15.self_attn.o_proj` | 312.5392 | 356.2232 |
| `model.layers.15.mlp.gate_proj` | 312.5392 | 330.7723 |
| `model.layers.15.mlp.up_proj` | 312.5392 | 325.2821 |
| `model.layers.15.mlp.down_proj` | 312.5392 | 343.0709 |
| `model.layers.16.self_attn.q_proj` | 312.5392 | 311.8786 |
| `model.layers.16.self_attn.k_proj` | 312.5392 | 314.7818 |
| `model.layers.16.self_attn.v_proj` | 312.5392 | 338.7047 |
| `model.layers.16.self_attn.o_proj` | 312.5392 | 347.8226 |
| `model.layers.16.mlp.gate_proj` | 312.5392 | 336.3187 |
| `model.layers.16.mlp.up_proj` | 312.5392 | 314.6022 |
| `model.layers.16.mlp.down_proj` | 312.5392 | 368.5835 |
| `model.layers.17.self_attn.q_proj` | 312.5392 | 314.3424 |
| `model.layers.17.self_attn.k_proj` | 312.5392 | 319.6807 |
| `model.layers.17.self_attn.v_proj` | 312.5392 | 349.3538 |
| `model.layers.17.self_attn.o_proj` | 312.5392 | 348.2813 |
| `model.layers.17.mlp.gate_proj` | 312.5392 | 326.5928 |
| `model.layers.17.mlp.up_proj` | 312.5392 | 324.8428 |
| `model.layers.17.mlp.down_proj` | 312.5392 | 343.2702 |
| `model.layers.18.self_attn.q_proj` | 312.5392 | 316.2140 |
| `model.layers.18.self_attn.k_proj` | 312.5392 | 316.6085 |
| `model.layers.18.self_attn.v_proj` | 312.5392 | 342.8209 |
| `model.layers.18.self_attn.o_proj` | 312.5392 | 338.6973 |
| `model.layers.18.mlp.gate_proj` | 312.5392 | 324.2102 |
| `model.layers.18.mlp.up_proj` | 312.5392 | 323.4813 |
| `model.layers.18.mlp.down_proj` | 312.5392 | 346.9024 |
| `model.layers.19.self_attn.q_proj` | 312.5392 | 314.8598 |
| `model.layers.19.self_attn.k_proj` | 312.5392 | 314.3897 |
| `model.layers.19.self_attn.v_proj` | 312.5392 | 339.4429 |
| `model.layers.19.self_attn.o_proj` | 312.5392 | 336.8599 |
| `model.layers.19.mlp.gate_proj` | 312.5392 | 313.9066 |
| `model.layers.19.mlp.up_proj` | 312.5392 | 323.2712 |
| `model.layers.19.mlp.down_proj` | 312.5392 | 340.4979 |
| `model.layers.20.self_attn.q_proj` | 312.5392 | 314.5818 |
| `model.layers.20.self_attn.k_proj` | 312.5392 | 313.9581 |
| `model.layers.20.self_attn.v_proj` | 312.5392 | 335.6040 |
| `model.layers.20.self_attn.o_proj` | 312.5392 | 330.3868 |
| `model.layers.20.mlp.gate_proj` | 312.5392 | 331.4390 |
| `model.layers.20.mlp.up_proj` | 312.5392 | 322.4181 |
| `model.layers.20.mlp.down_proj` | 312.5392 | 351.5521 |
| `model.layers.21.self_attn.q_proj` | 312.5392 | 314.1204 |
| `model.layers.21.self_attn.k_proj` | 312.5392 | 314.2747 |
| `model.layers.21.self_attn.v_proj` | 312.5392 | 349.3407 |
| `model.layers.21.self_attn.o_proj` | 312.5392 | 339.7783 |
| `model.layers.21.mlp.gate_proj` | 312.5392 | 318.7106 |
| `model.layers.21.mlp.up_proj` | 312.5392 | 318.1945 |
| `model.layers.21.mlp.down_proj` | 312.5392 | 338.7282 |
| `model.layers.22.self_attn.q_proj` | 312.5392 | 311.6155 |
| `model.layers.22.self_attn.k_proj` | 312.5392 | 313.1237 |
| `model.layers.22.self_attn.v_proj` | 312.5392 | 313.0033 |
| `model.layers.22.self_attn.o_proj` | 312.5392 | 321.4512 |
| `model.layers.22.mlp.gate_proj` | 312.5392 | 318.1651 |
| `model.layers.22.mlp.up_proj` | 312.5392 | 324.7101 |
| `model.layers.22.mlp.down_proj` | 312.5392 | 352.9224 |
| `model.layers.23.self_attn.q_proj` | 312.5392 | 312.9730 |
| `model.layers.23.self_attn.k_proj` | 312.5392 | 312.1875 |
| `model.layers.23.self_attn.v_proj` | 312.5392 | 328.2809 |
| `model.layers.23.self_attn.o_proj` | 312.5392 | 320.8591 |
| `model.layers.23.mlp.gate_proj` | 312.5392 | 309.6038 |
| `model.layers.23.mlp.up_proj` | 312.5392 | 304.8419 |
| `model.layers.23.mlp.down_proj` | 312.5392 | 321.0719 |
| `model.layers.24.self_attn.q_proj` | 312.5392 | 313.3277 |
| `model.layers.24.self_attn.k_proj` | 312.5392 | 313.6618 |
| `model.layers.24.self_attn.v_proj` | 312.5392 | 318.3073 |
| `model.layers.24.self_attn.o_proj` | 312.5392 | 325.6475 |
| `model.layers.24.mlp.gate_proj` | 312.5392 | 319.1357 |
| `model.layers.24.mlp.up_proj` | 312.5392 | 319.0833 |
| `model.layers.24.mlp.down_proj` | 312.5392 | 327.8884 |
| `model.layers.25.self_attn.q_proj` | 312.5392 | 312.7742 |
| `model.layers.25.self_attn.k_proj` | 312.5392 | 311.9422 |
| `model.layers.25.self_attn.v_proj` | 312.5392 | 341.0506 |
| `model.layers.25.self_attn.o_proj` | 312.5392 | 327.6587 |
| `model.layers.25.mlp.gate_proj` | 312.5392 | 318.2362 |
| `model.layers.25.mlp.up_proj` | 312.5392 | 313.1918 |
| `model.layers.25.mlp.down_proj` | 312.5392 | 332.3494 |
| `model.layers.26.self_attn.q_proj` | 312.5392 | 311.8159 |
| `model.layers.26.self_attn.k_proj` | 312.5392 | 313.2718 |
| `model.layers.26.self_attn.v_proj` | 312.5392 | 357.0975 |
| `model.layers.26.self_attn.o_proj` | 312.5392 | 338.7629 |
| `model.layers.26.mlp.gate_proj` | 312.5392 | 327.4417 |
| `model.layers.26.mlp.up_proj` | 312.5392 | 314.7103 |
| `model.layers.26.mlp.down_proj` | 312.5392 | 324.6607 |
| `model.layers.27.self_attn.q_proj` | 312.5392 | 309.4084 |
| `model.layers.27.self_attn.k_proj` | 312.5392 | 310.7245 |
| `model.layers.27.self_attn.v_proj` | 312.5392 | 323.8905 |
| `model.layers.27.self_attn.o_proj` | 312.5392 | 318.3723 |
| `model.layers.27.mlp.gate_proj` | 312.5392 | 318.1355 |
| `model.layers.27.mlp.up_proj` | 312.5392 | 314.3954 |
| `model.layers.27.mlp.down_proj` | 312.5392 | 335.1273 |
| `model.layers.28.self_attn.q_proj` | 312.5392 | 313.2295 |
| `model.layers.28.self_attn.k_proj` | 312.5392 | 313.5040 |
| `model.layers.28.self_attn.v_proj` | 312.5392 | 343.4230 |
| `model.layers.28.self_attn.o_proj` | 312.5392 | 330.3073 |
| `model.layers.28.mlp.gate_proj` | 312.5392 | 331.4600 |
| `model.layers.28.mlp.up_proj` | 312.5392 | 322.8607 |
| `model.layers.28.mlp.down_proj` | 312.5392 | 334.1450 |
| `model.layers.29.self_attn.q_proj` | 312.5392 | 311.2898 |
| `model.layers.29.self_attn.k_proj` | 312.5392 | 314.0581 |
| `model.layers.29.self_attn.v_proj` | 312.5392 | 327.4563 |
| `model.layers.29.self_attn.o_proj` | 312.5392 | 330.4463 |
| `model.layers.29.mlp.gate_proj` | 312.5392 | 330.9421 |
| `model.layers.29.mlp.up_proj` | 312.5392 | 311.7183 |
| `model.layers.29.mlp.down_proj` | 312.5392 | 332.8676 |
| `model.layers.30.self_attn.q_proj` | 312.5392 | 314.3447 |
| `model.layers.30.self_attn.k_proj` | 312.5392 | 315.6959 |
| `model.layers.30.self_attn.v_proj` | 312.5392 | 342.0208 |
| `model.layers.30.self_attn.o_proj` | 312.5392 | 325.9998 |
| `model.layers.30.mlp.gate_proj` | 312.5392 | 313.2512 |
| `model.layers.30.mlp.up_proj` | 312.5392 | 317.8566 |
| `model.layers.30.mlp.down_proj` | 312.5392 | 343.6867 |
| `model.layers.31.self_attn.q_proj` | 312.5392 | 313.0353 |
| `model.layers.31.self_attn.k_proj` | 312.5392 | 314.3104 |
| `model.layers.31.self_attn.v_proj` | 312.5392 | 325.4233 |
| `model.layers.31.self_attn.o_proj` | 312.5392 | 335.7356 |
| `model.layers.31.mlp.gate_proj` | 312.5392 | 323.8217 |
| `model.layers.31.mlp.up_proj` | 312.5392 | 323.7858 |
| `model.layers.31.mlp.down_proj` | 312.5392 | 407.9331 |

## Loss Delta Grid

| Layer | baseline | 8192 |
| --- | --- | --- |
| `model.layers.0.self_attn.q_proj` | 0.0000 | 0.0366 |
| `model.layers.0.self_attn.k_proj` | 0.0000 | 0.0007 |
| `model.layers.0.self_attn.v_proj` | 0.0000 | 5.5832 |
| `model.layers.0.self_attn.o_proj` | 0.0000 | 0.1495 |
| `model.layers.0.mlp.gate_proj` | 0.0000 | -0.2067 |
| `model.layers.0.mlp.up_proj` | 0.0000 | 0.0919 |
| `model.layers.0.mlp.down_proj` | 0.0000 | 3.1524 |
| `model.layers.1.self_attn.q_proj` | 0.0000 | -0.0081 |
| `model.layers.1.self_attn.k_proj` | 0.0000 | 0.0402 |
| `model.layers.1.self_attn.v_proj` | 0.0000 | 4.9969 |
| `model.layers.1.self_attn.o_proj` | 0.0000 | 0.3908 |
| `model.layers.1.mlp.gate_proj` | 0.0000 | -0.1139 |
| `model.layers.1.mlp.up_proj` | 0.0000 | 0.0811 |
| `model.layers.1.mlp.down_proj` | 0.0000 | 5.5121 |
| `model.layers.2.self_attn.q_proj` | 0.0000 | 0.0143 |
| `model.layers.2.self_attn.k_proj` | 0.0000 | 0.0016 |
| `model.layers.2.self_attn.v_proj` | 0.0000 | 2.4299 |
| `model.layers.2.self_attn.o_proj` | 0.0000 | 0.3559 |
| `model.layers.2.mlp.gate_proj` | 0.0000 | 0.1166 |
| `model.layers.2.mlp.up_proj` | 0.0000 | -0.0187 |
| `model.layers.2.mlp.down_proj` | 0.0000 | 0.3535 |
| `model.layers.3.self_attn.q_proj` | 0.0000 | 0.0065 |
| `model.layers.3.self_attn.k_proj` | 0.0000 | 0.0413 |
| `model.layers.3.self_attn.v_proj` | 0.0000 | 2.4242 |
| `model.layers.3.self_attn.o_proj` | 0.0000 | 0.2119 |
| `model.layers.3.mlp.gate_proj` | 0.0000 | 0.1243 |
| `model.layers.3.mlp.up_proj` | 0.0000 | 0.1632 |
| `model.layers.3.mlp.down_proj` | 0.0000 | 0.4466 |
| `model.layers.4.self_attn.q_proj` | 0.0000 | 0.0098 |
| `model.layers.4.self_attn.k_proj` | 0.0000 | 0.0771 |
| `model.layers.4.self_attn.v_proj` | 0.0000 | 5.2200 |
| `model.layers.4.self_attn.o_proj` | 0.0000 | 0.2354 |
| `model.layers.4.mlp.gate_proj` | 0.0000 | 0.1578 |
| `model.layers.4.mlp.up_proj` | 0.0000 | 0.1612 |
| `model.layers.4.mlp.down_proj` | 0.0000 | 0.7188 |
| `model.layers.5.self_attn.q_proj` | 0.0000 | -0.0047 |
| `model.layers.5.self_attn.k_proj` | 0.0000 | 0.0081 |
| `model.layers.5.self_attn.v_proj` | 0.0000 | 0.9069 |
| `model.layers.5.self_attn.o_proj` | 0.0000 | 0.3126 |
| `model.layers.5.mlp.gate_proj` | 0.0000 | 0.0820 |
| `model.layers.5.mlp.up_proj` | 0.0000 | 0.0731 |
| `model.layers.5.mlp.down_proj` | 0.0000 | 0.6884 |
| `model.layers.6.self_attn.q_proj` | 0.0000 | 0.0278 |
| `model.layers.6.self_attn.k_proj` | 0.0000 | -0.0008 |
| `model.layers.6.self_attn.v_proj` | 0.0000 | 0.5200 |
| `model.layers.6.self_attn.o_proj` | 0.0000 | 0.1031 |
| `model.layers.6.mlp.gate_proj` | 0.0000 | 0.0380 |
| `model.layers.6.mlp.up_proj` | 0.0000 | 0.1054 |
| `model.layers.6.mlp.down_proj` | 0.0000 | 0.4478 |
| `model.layers.7.self_attn.q_proj` | 0.0000 | 0.0074 |
| `model.layers.7.self_attn.k_proj` | 0.0000 | 0.0188 |
| `model.layers.7.self_attn.v_proj` | 0.0000 | 0.4519 |
| `model.layers.7.self_attn.o_proj` | 0.0000 | 0.2278 |
| `model.layers.7.mlp.gate_proj` | 0.0000 | 0.0632 |
| `model.layers.7.mlp.up_proj` | 0.0000 | 0.0335 |
| `model.layers.7.mlp.down_proj` | 0.0000 | 0.2346 |
| `model.layers.8.self_attn.q_proj` | 0.0000 | 0.0269 |
| `model.layers.8.self_attn.k_proj` | 0.0000 | 0.0162 |
| `model.layers.8.self_attn.v_proj` | 0.0000 | 0.3482 |
| `model.layers.8.self_attn.o_proj` | 0.0000 | 0.0987 |
| `model.layers.8.mlp.gate_proj` | 0.0000 | 0.0592 |
| `model.layers.8.mlp.up_proj` | 0.0000 | 0.1106 |
| `model.layers.8.mlp.down_proj` | 0.0000 | 0.2660 |
| `model.layers.9.self_attn.q_proj` | 0.0000 | 0.0162 |
| `model.layers.9.self_attn.k_proj` | 0.0000 | 0.0037 |
| `model.layers.9.self_attn.v_proj` | 0.0000 | 0.3141 |
| `model.layers.9.self_attn.o_proj` | 0.0000 | 0.0704 |
| `model.layers.9.mlp.gate_proj` | 0.0000 | 0.0940 |
| `model.layers.9.mlp.up_proj` | 0.0000 | 0.0873 |
| `model.layers.9.mlp.down_proj` | 0.0000 | 0.3817 |
| `model.layers.10.self_attn.q_proj` | 0.0000 | 0.0096 |
| `model.layers.10.self_attn.k_proj` | 0.0000 | 0.0147 |
| `model.layers.10.self_attn.v_proj` | 0.0000 | 0.1106 |
| `model.layers.10.self_attn.o_proj` | 0.0000 | 0.1931 |
| `model.layers.10.mlp.gate_proj` | 0.0000 | 0.0655 |
| `model.layers.10.mlp.up_proj` | 0.0000 | 0.0312 |
| `model.layers.10.mlp.down_proj` | 0.0000 | 0.2474 |
| `model.layers.11.self_attn.q_proj` | 0.0000 | 0.0207 |
| `model.layers.11.self_attn.k_proj` | 0.0000 | 0.0047 |
| `model.layers.11.self_attn.v_proj` | 0.0000 | 0.2996 |
| `model.layers.11.self_attn.o_proj` | 0.0000 | 0.1774 |
| `model.layers.11.mlp.gate_proj` | 0.0000 | 0.0398 |
| `model.layers.11.mlp.up_proj` | 0.0000 | 0.0148 |
| `model.layers.11.mlp.down_proj` | 0.0000 | 0.1559 |
| `model.layers.12.self_attn.q_proj` | 0.0000 | 0.0483 |
| `model.layers.12.self_attn.k_proj` | 0.0000 | 0.0266 |
| `model.layers.12.self_attn.v_proj` | 0.0000 | 0.2340 |
| `model.layers.12.self_attn.o_proj` | 0.0000 | 0.1255 |
| `model.layers.12.mlp.gate_proj` | 0.0000 | 0.0187 |
| `model.layers.12.mlp.up_proj` | 0.0000 | 0.0208 |
| `model.layers.12.mlp.down_proj` | 0.0000 | 0.1357 |
| `model.layers.13.self_attn.q_proj` | 0.0000 | 0.0163 |
| `model.layers.13.self_attn.k_proj` | 0.0000 | -0.0029 |
| `model.layers.13.self_attn.v_proj` | 0.0000 | 0.1830 |
| `model.layers.13.self_attn.o_proj` | 0.0000 | 0.1876 |
| `model.layers.13.mlp.gate_proj` | 0.0000 | 0.0435 |
| `model.layers.13.mlp.up_proj` | 0.0000 | 0.0891 |
| `model.layers.13.mlp.down_proj` | 0.0000 | 0.1652 |
| `model.layers.14.self_attn.q_proj` | 0.0000 | -0.0070 |
| `model.layers.14.self_attn.k_proj` | 0.0000 | 0.0141 |
| `model.layers.14.self_attn.v_proj` | 0.0000 | 0.1775 |
| `model.layers.14.self_attn.o_proj` | 0.0000 | 0.0913 |
| `model.layers.14.mlp.gate_proj` | 0.0000 | 0.0244 |
| `model.layers.14.mlp.up_proj` | 0.0000 | 0.0292 |
| `model.layers.14.mlp.down_proj` | 0.0000 | 0.1512 |
| `model.layers.15.self_attn.q_proj` | 0.0000 | 0.0053 |
| `model.layers.15.self_attn.k_proj` | 0.0000 | 0.0068 |
| `model.layers.15.self_attn.v_proj` | 0.0000 | 0.2470 |
| `model.layers.15.self_attn.o_proj` | 0.0000 | 0.1308 |
| `model.layers.15.mlp.gate_proj` | 0.0000 | 0.0567 |
| `model.layers.15.mlp.up_proj` | 0.0000 | 0.0400 |
| `model.layers.15.mlp.down_proj` | 0.0000 | 0.0932 |
| `model.layers.16.self_attn.q_proj` | 0.0000 | -0.0021 |
| `model.layers.16.self_attn.k_proj` | 0.0000 | 0.0071 |
| `model.layers.16.self_attn.v_proj` | 0.0000 | 0.0804 |
| `model.layers.16.self_attn.o_proj` | 0.0000 | 0.1070 |
| `model.layers.16.mlp.gate_proj` | 0.0000 | 0.0733 |
| `model.layers.16.mlp.up_proj` | 0.0000 | 0.0066 |
| `model.layers.16.mlp.down_proj` | 0.0000 | 0.1649 |
| `model.layers.17.self_attn.q_proj` | 0.0000 | 0.0058 |
| `model.layers.17.self_attn.k_proj` | 0.0000 | 0.0226 |
| `model.layers.17.self_attn.v_proj` | 0.0000 | 0.1114 |
| `model.layers.17.self_attn.o_proj` | 0.0000 | 0.1083 |
| `model.layers.17.mlp.gate_proj` | 0.0000 | 0.0440 |
| `model.layers.17.mlp.up_proj` | 0.0000 | 0.0386 |
| `model.layers.17.mlp.down_proj` | 0.0000 | 0.0938 |
| `model.layers.18.self_attn.q_proj` | 0.0000 | 0.0117 |
| `model.layers.18.self_attn.k_proj` | 0.0000 | 0.0129 |
| `model.layers.18.self_attn.v_proj` | 0.0000 | 0.0925 |
| `model.layers.18.self_attn.o_proj` | 0.0000 | 0.0804 |
| `model.layers.18.mlp.gate_proj` | 0.0000 | 0.0367 |
| `model.layers.18.mlp.up_proj` | 0.0000 | 0.0344 |
| `model.layers.18.mlp.down_proj` | 0.0000 | 0.1043 |
| `model.layers.19.self_attn.q_proj` | 0.0000 | 0.0074 |
| `model.layers.19.self_attn.k_proj` | 0.0000 | 0.0059 |
| `model.layers.19.self_attn.v_proj` | 0.0000 | 0.0826 |
| `model.layers.19.self_attn.o_proj` | 0.0000 | 0.0749 |
| `model.layers.19.mlp.gate_proj` | 0.0000 | 0.0044 |
| `model.layers.19.mlp.up_proj` | 0.0000 | 0.0338 |
| `model.layers.19.mlp.down_proj` | 0.0000 | 0.0857 |
| `model.layers.20.self_attn.q_proj` | 0.0000 | 0.0065 |
| `model.layers.20.self_attn.k_proj` | 0.0000 | 0.0045 |
| `model.layers.20.self_attn.v_proj` | 0.0000 | 0.0712 |
| `model.layers.20.self_attn.o_proj` | 0.0000 | 0.0555 |
| `model.layers.20.mlp.gate_proj` | 0.0000 | 0.0587 |
| `model.layers.20.mlp.up_proj` | 0.0000 | 0.0311 |
| `model.layers.20.mlp.down_proj` | 0.0000 | 0.1176 |
| `model.layers.21.self_attn.q_proj` | 0.0000 | 0.0050 |
| `model.layers.21.self_attn.k_proj` | 0.0000 | 0.0055 |
| `model.layers.21.self_attn.v_proj` | 0.0000 | 0.1113 |
| `model.layers.21.self_attn.o_proj` | 0.0000 | 0.0836 |
| `model.layers.21.mlp.gate_proj` | 0.0000 | 0.0196 |
| `model.layers.21.mlp.up_proj` | 0.0000 | 0.0179 |
| `model.layers.21.mlp.down_proj` | 0.0000 | 0.0805 |
| `model.layers.22.self_attn.q_proj` | 0.0000 | -0.0030 |
| `model.layers.22.self_attn.k_proj` | 0.0000 | 0.0019 |
| `model.layers.22.self_attn.v_proj` | 0.0000 | 0.0015 |
| `model.layers.22.self_attn.o_proj` | 0.0000 | 0.0281 |
| `model.layers.22.mlp.gate_proj` | 0.0000 | 0.0178 |
| `model.layers.22.mlp.up_proj` | 0.0000 | 0.0382 |
| `model.layers.22.mlp.down_proj` | 0.0000 | 0.1215 |
| `model.layers.23.self_attn.q_proj` | 0.0000 | 0.0014 |
| `model.layers.23.self_attn.k_proj` | 0.0000 | -0.0011 |
| `model.layers.23.self_attn.v_proj` | 0.0000 | 0.0491 |
| `model.layers.23.self_attn.o_proj` | 0.0000 | 0.0263 |
| `model.layers.23.mlp.gate_proj` | 0.0000 | -0.0094 |
| `model.layers.23.mlp.up_proj` | 0.0000 | -0.0249 |
| `model.layers.23.mlp.down_proj` | 0.0000 | 0.0269 |
| `model.layers.24.self_attn.q_proj` | 0.0000 | 0.0025 |
| `model.layers.24.self_attn.k_proj` | 0.0000 | 0.0036 |
| `model.layers.24.self_attn.v_proj` | 0.0000 | 0.0183 |
| `model.layers.24.self_attn.o_proj` | 0.0000 | 0.0411 |
| `model.layers.24.mlp.gate_proj` | 0.0000 | 0.0209 |
| `model.layers.24.mlp.up_proj` | 0.0000 | 0.0207 |
| `model.layers.24.mlp.down_proj` | 0.0000 | 0.0479 |
| `model.layers.25.self_attn.q_proj` | 0.0000 | 0.0008 |
| `model.layers.25.self_attn.k_proj` | 0.0000 | -0.0019 |
| `model.layers.25.self_attn.v_proj` | 0.0000 | 0.0873 |
| `model.layers.25.self_attn.o_proj` | 0.0000 | 0.0472 |
| `model.layers.25.mlp.gate_proj` | 0.0000 | 0.0181 |
| `model.layers.25.mlp.up_proj` | 0.0000 | 0.0021 |
| `model.layers.25.mlp.down_proj` | 0.0000 | 0.0615 |
| `model.layers.26.self_attn.q_proj` | 0.0000 | -0.0023 |
| `model.layers.26.self_attn.k_proj` | 0.0000 | 0.0023 |
| `model.layers.26.self_attn.v_proj` | 0.0000 | 0.1333 |
| `model.layers.26.self_attn.o_proj` | 0.0000 | 0.0806 |
| `model.layers.26.mlp.gate_proj` | 0.0000 | 0.0466 |
| `model.layers.26.mlp.up_proj` | 0.0000 | 0.0069 |
| `model.layers.26.mlp.down_proj` | 0.0000 | 0.0381 |
| `model.layers.27.self_attn.q_proj` | 0.0000 | -0.0101 |
| `model.layers.27.self_attn.k_proj` | 0.0000 | -0.0058 |
| `model.layers.27.self_attn.v_proj` | 0.0000 | 0.0357 |
| `model.layers.27.self_attn.o_proj` | 0.0000 | 0.0185 |
| `model.layers.27.mlp.gate_proj` | 0.0000 | 0.0177 |
| `model.layers.27.mlp.up_proj` | 0.0000 | 0.0059 |
| `model.layers.27.mlp.down_proj` | 0.0000 | 0.0698 |
| `model.layers.28.self_attn.q_proj` | 0.0000 | 0.0022 |
| `model.layers.28.self_attn.k_proj` | 0.0000 | 0.0031 |
| `model.layers.28.self_attn.v_proj` | 0.0000 | 0.0942 |
| `model.layers.28.self_attn.o_proj` | 0.0000 | 0.0553 |
| `model.layers.28.mlp.gate_proj` | 0.0000 | 0.0588 |
| `model.layers.28.mlp.up_proj` | 0.0000 | 0.0325 |
| `model.layers.28.mlp.down_proj` | 0.0000 | 0.0668 |
| `model.layers.29.self_attn.q_proj` | 0.0000 | -0.0040 |
| `model.layers.29.self_attn.k_proj` | 0.0000 | 0.0048 |
| `model.layers.29.self_attn.v_proj` | 0.0000 | 0.0466 |
| `model.layers.29.self_attn.o_proj` | 0.0000 | 0.0557 |
| `model.layers.29.mlp.gate_proj` | 0.0000 | 0.0572 |
| `model.layers.29.mlp.up_proj` | 0.0000 | -0.0026 |
| `model.layers.29.mlp.down_proj` | 0.0000 | 0.0630 |
| `model.layers.30.self_attn.q_proj` | 0.0000 | 0.0058 |
| `model.layers.30.self_attn.k_proj` | 0.0000 | 0.0100 |
| `model.layers.30.self_attn.v_proj` | 0.0000 | 0.0901 |
| `model.layers.30.self_attn.o_proj` | 0.0000 | 0.0422 |
| `model.layers.30.mlp.gate_proj` | 0.0000 | 0.0023 |
| `model.layers.30.mlp.up_proj` | 0.0000 | 0.0169 |
| `model.layers.30.mlp.down_proj` | 0.0000 | 0.0950 |
| `model.layers.31.self_attn.q_proj` | 0.0000 | 0.0016 |
| `model.layers.31.self_attn.k_proj` | 0.0000 | 0.0057 |
| `model.layers.31.self_attn.v_proj` | 0.0000 | 0.0404 |
| `model.layers.31.self_attn.o_proj` | 0.0000 | 0.0716 |
| `model.layers.31.mlp.gate_proj` | 0.0000 | 0.0355 |
| `model.layers.31.mlp.up_proj` | 0.0000 | 0.0354 |
| `model.layers.31.mlp.down_proj` | 0.0000 | 0.2664 |

## Layer Metadata

| Layer | Weight shape | Best k | Best ppl | Worst k | Worst ppl |
| --- | --- | ---: | ---: | ---: | ---: |
| `model.layers.0.self_attn.q_proj` | `(4096, 4096)` | 8192 | 324.1992 | 8192 | 324.1992 |
| `model.layers.0.self_attn.k_proj` | `(1024, 4096)` | 8192 | 312.7727 | 8192 | 312.7727 |
| `model.layers.0.self_attn.v_proj` | `(1024, 4096)` | 8192 | 83114.8267 | 8192 | 83114.8267 |
| `model.layers.0.self_attn.o_proj` | `(4096, 4096)` | 8192 | 362.9385 | 8192 | 362.9385 |
| `model.layers.0.mlp.gate_proj` | `(14336, 4096)` | 8192 | 254.1718 | 8192 | 254.1718 |
| `model.layers.0.mlp.up_proj` | `(14336, 4096)` | 8192 | 342.6199 | 8192 | 342.6199 |
| `model.layers.0.mlp.down_proj` | `(4096, 14336)` | 8192 | 7310.7251 | 8192 | 7310.7251 |
| `model.layers.1.self_attn.q_proj` | `(4096, 4096)` | 8192 | 310.0183 | 8192 | 310.0183 |
| `model.layers.1.self_attn.k_proj` | `(1024, 4096)` | 8192 | 325.3677 | 8192 | 325.3677 |
| `model.layers.1.self_attn.v_proj` | `(1024, 4096)` | 8192 | 46241.1073 | 8192 | 46241.1073 |
| `model.layers.1.self_attn.o_proj` | `(4096, 4096)` | 8192 | 462.0041 | 8192 | 462.0041 |
| `model.layers.1.mlp.gate_proj` | `(14336, 4096)` | 8192 | 278.8959 | 8192 | 278.8959 |
| `model.layers.1.mlp.up_proj` | `(14336, 4096)` | 8192 | 338.9521 | 8192 | 338.9521 |
| `model.layers.1.mlp.down_proj` | `(4096, 14336)` | 8192 | 77406.6199 | 8192 | 77406.6199 |
| `model.layers.2.self_attn.q_proj` | `(4096, 4096)` | 8192 | 317.0302 | 8192 | 317.0302 |
| `model.layers.2.self_attn.k_proj` | `(1024, 4096)` | 8192 | 313.0373 | 8192 | 313.0373 |
| `model.layers.2.self_attn.v_proj` | `(1024, 4096)` | 8192 | 3549.6047 | 8192 | 3549.6047 |
| `model.layers.2.self_attn.o_proj` | `(4096, 4096)` | 8192 | 446.1478 | 8192 | 446.1478 |
| `model.layers.2.mlp.gate_proj` | `(14336, 4096)` | 8192 | 351.1852 | 8192 | 351.1852 |
| `model.layers.2.mlp.up_proj` | `(14336, 4096)` | 8192 | 306.7409 | 8192 | 306.7409 |
| `model.layers.2.mlp.down_proj` | `(4096, 14336)` | 8192 | 445.0670 | 8192 | 445.0670 |
| `model.layers.3.self_attn.q_proj` | `(4096, 4096)` | 8192 | 314.5859 | 8192 | 314.5859 |
| `model.layers.3.self_attn.k_proj` | `(1024, 4096)` | 8192 | 325.7234 | 8192 | 325.7234 |
| `model.layers.3.self_attn.v_proj` | `(1024, 4096)` | 8192 | 3529.4682 | 8192 | 3529.4682 |
| `model.layers.3.self_attn.o_proj` | `(4096, 4096)` | 8192 | 386.2949 | 8192 | 386.2949 |
| `model.layers.3.mlp.gate_proj` | `(14336, 4096)` | 8192 | 353.9159 | 8192 | 353.9159 |
| `model.layers.3.mlp.up_proj` | `(14336, 4096)` | 8192 | 367.9400 | 8192 | 367.9400 |
| `model.layers.3.mlp.down_proj` | `(4096, 14336)` | 8192 | 488.5048 | 8192 | 488.5048 |
| `model.layers.4.self_attn.q_proj` | `(4096, 4096)` | 8192 | 315.6097 | 8192 | 315.6097 |
| `model.layers.4.self_attn.k_proj` | `(1024, 4096)` | 8192 | 337.5957 | 8192 | 337.5957 |
| `model.layers.4.self_attn.v_proj` | `(1024, 4096)` | 8192 | 57800.1001 | 8192 | 57800.1001 |
| `model.layers.4.self_attn.o_proj` | `(4096, 4096)` | 8192 | 395.5000 | 8192 | 395.5000 |
| `model.layers.4.mlp.gate_proj` | `(14336, 4096)` | 8192 | 365.9487 | 8192 | 365.9487 |
| `model.layers.4.mlp.up_proj` | `(14336, 4096)` | 8192 | 367.2004 | 8192 | 367.2004 |
| `model.layers.4.mlp.down_proj` | `(4096, 14336)` | 8192 | 641.3038 | 8192 | 641.3038 |
| `model.layers.5.self_attn.q_proj` | `(4096, 4096)` | 8192 | 311.0822 | 8192 | 311.0822 |
| `model.layers.5.self_attn.k_proj` | `(1024, 4096)` | 8192 | 315.0707 | 8192 | 315.0707 |
| `model.layers.5.self_attn.v_proj` | `(1024, 4096)` | 8192 | 774.0688 | 8192 | 774.0688 |
| `model.layers.5.self_attn.o_proj` | `(4096, 4096)` | 8192 | 427.2294 | 8192 | 427.2294 |
| `model.layers.5.mlp.gate_proj` | `(14336, 4096)` | 8192 | 339.2610 | 8192 | 339.2610 |
| `model.layers.5.mlp.up_proj` | `(14336, 4096)` | 8192 | 336.2553 | 8192 | 336.2553 |
| `model.layers.5.mlp.down_proj` | `(4096, 14336)` | 8192 | 622.1316 | 8192 | 622.1316 |
| `model.layers.6.self_attn.q_proj` | `(4096, 4096)` | 8192 | 321.3569 | 8192 | 321.3569 |
| `model.layers.6.self_attn.k_proj` | `(1024, 4096)` | 8192 | 312.2848 | 8192 | 312.2848 |
| `model.layers.6.self_attn.v_proj` | `(1024, 4096)` | 8192 | 525.7204 | 8192 | 525.7204 |
| `model.layers.6.self_attn.o_proj` | `(4096, 4096)` | 8192 | 346.4862 | 8192 | 346.4862 |
| `model.layers.6.mlp.gate_proj` | `(14336, 4096)` | 8192 | 324.6501 | 8192 | 324.6501 |
| `model.layers.6.mlp.up_proj` | `(14336, 4096)` | 8192 | 347.2855 | 8192 | 347.2855 |
| `model.layers.6.mlp.down_proj` | `(4096, 14336)` | 8192 | 489.0584 | 8192 | 489.0584 |
| `model.layers.7.self_attn.q_proj` | `(4096, 4096)` | 8192 | 314.8727 | 8192 | 314.8727 |
| `model.layers.7.self_attn.k_proj` | `(1024, 4096)` | 8192 | 318.4802 | 8192 | 318.4802 |
| `model.layers.7.self_attn.v_proj` | `(1024, 4096)` | 8192 | 491.0802 | 8192 | 491.0802 |
| `model.layers.7.self_attn.o_proj` | `(4096, 4096)` | 8192 | 392.5139 | 8192 | 392.5139 |
| `model.layers.7.mlp.gate_proj` | `(14336, 4096)` | 8192 | 332.9240 | 8192 | 332.9240 |
| `model.layers.7.mlp.up_proj` | `(14336, 4096)` | 8192 | 323.1782 | 8192 | 323.1782 |
| `model.layers.7.mlp.down_proj` | `(4096, 14336)` | 8192 | 395.1580 | 8192 | 395.1580 |
| `model.layers.8.self_attn.q_proj` | `(4096, 4096)` | 8192 | 321.0604 | 8192 | 321.0604 |
| `model.layers.8.self_attn.k_proj` | `(1024, 4096)` | 8192 | 317.6524 | 8192 | 317.6524 |
| `model.layers.8.self_attn.v_proj` | `(1024, 4096)` | 8192 | 442.7207 | 8192 | 442.7207 |
| `model.layers.8.self_attn.o_proj` | `(4096, 4096)` | 8192 | 344.9528 | 8192 | 344.9528 |
| `model.layers.8.mlp.gate_proj` | `(14336, 4096)` | 8192 | 331.5877 | 8192 | 331.5877 |
| `model.layers.8.mlp.up_proj` | `(14336, 4096)` | 8192 | 349.0940 | 8192 | 349.0940 |
| `model.layers.8.mlp.down_proj` | `(4096, 14336)` | 8192 | 407.7899 | 8192 | 407.7899 |
| `model.layers.9.self_attn.q_proj` | `(4096, 4096)` | 8192 | 317.6568 | 8192 | 317.6568 |
| `model.layers.9.self_attn.k_proj` | `(1024, 4096)` | 8192 | 313.6879 | 8192 | 313.6879 |
| `model.layers.9.self_attn.v_proj` | `(1024, 4096)` | 8192 | 427.8800 | 8192 | 427.8800 |
| `model.layers.9.self_attn.o_proj` | `(4096, 4096)` | 8192 | 335.3298 | 8192 | 335.3298 |
| `model.layers.9.mlp.gate_proj` | `(14336, 4096)` | 8192 | 343.3493 | 8192 | 343.3493 |
| `model.layers.9.mlp.up_proj` | `(14336, 4096)` | 8192 | 341.0644 | 8192 | 341.0644 |
| `model.layers.9.mlp.down_proj` | `(4096, 14336)` | 8192 | 457.8151 | 8192 | 457.8151 |
| `model.layers.10.self_attn.q_proj` | `(4096, 4096)` | 8192 | 315.5471 | 8192 | 315.5471 |
| `model.layers.10.self_attn.k_proj` | `(1024, 4096)` | 8192 | 317.1745 | 8192 | 317.1745 |
| `model.layers.10.self_attn.v_proj` | `(1024, 4096)` | 8192 | 349.1036 | 8192 | 349.1036 |
| `model.layers.10.self_attn.o_proj` | `(4096, 4096)` | 8192 | 379.1018 | 8192 | 379.1018 |
| `model.layers.10.mlp.gate_proj` | `(14336, 4096)` | 8192 | 333.6819 | 8192 | 333.6819 |
| `model.layers.10.mlp.up_proj` | `(14336, 4096)` | 8192 | 322.4343 | 8192 | 322.4343 |
| `model.layers.10.mlp.down_proj` | `(4096, 14336)` | 8192 | 400.2514 | 8192 | 400.2514 |
| `model.layers.11.self_attn.q_proj` | `(4096, 4096)` | 8192 | 319.0883 | 8192 | 319.0883 |
| `model.layers.11.self_attn.k_proj` | `(1024, 4096)` | 8192 | 314.0070 | 8192 | 314.0070 |
| `model.layers.11.self_attn.v_proj` | `(1024, 4096)` | 8192 | 421.7309 | 8192 | 421.7309 |
| `model.layers.11.self_attn.o_proj` | `(4096, 4096)` | 8192 | 373.2205 | 8192 | 373.2205 |
| `model.layers.11.mlp.gate_proj` | `(14336, 4096)` | 8192 | 325.2441 | 8192 | 325.2441 |
| `model.layers.11.mlp.up_proj` | `(14336, 4096)` | 8192 | 317.2083 | 8192 | 317.2083 |
| `model.layers.11.mlp.down_proj` | `(4096, 14336)` | 8192 | 365.2770 | 8192 | 365.2770 |
| `model.layers.12.self_attn.q_proj` | `(4096, 4096)` | 8192 | 327.9992 | 8192 | 327.9992 |
| `model.layers.12.self_attn.k_proj` | `(1024, 4096)` | 8192 | 320.9580 | 8192 | 320.9580 |
| `model.layers.12.self_attn.v_proj` | `(1024, 4096)` | 8192 | 394.9543 | 8192 | 394.9543 |
| `model.layers.12.self_attn.o_proj` | `(4096, 4096)` | 8192 | 354.3359 | 8192 | 354.3359 |
| `model.layers.12.mlp.gate_proj` | `(14336, 4096)` | 8192 | 318.4440 | 8192 | 318.4440 |
| `model.layers.12.mlp.up_proj` | `(14336, 4096)` | 8192 | 319.1065 | 8192 | 319.1065 |
| `model.layers.12.mlp.down_proj` | `(4096, 14336)` | 8192 | 357.9694 | 8192 | 357.9694 |
| `model.layers.13.self_attn.q_proj` | `(4096, 4096)` | 8192 | 317.6807 | 8192 | 317.6807 |
| `model.layers.13.self_attn.k_proj` | `(1024, 4096)` | 8192 | 311.6374 | 8192 | 311.6374 |
| `model.layers.13.self_attn.v_proj` | `(1024, 4096)` | 8192 | 375.3091 | 8192 | 375.3091 |
| `model.layers.13.self_attn.o_proj` | `(4096, 4096)` | 8192 | 377.0205 | 8192 | 377.0205 |
| `model.layers.13.mlp.gate_proj` | `(14336, 4096)` | 8192 | 326.4315 | 8192 | 326.4315 |
| `model.layers.13.mlp.up_proj` | `(14336, 4096)` | 8192 | 341.6754 | 8192 | 341.6754 |
| `model.layers.13.mlp.down_proj` | `(4096, 14336)` | 8192 | 368.6730 | 8192 | 368.6730 |
| `model.layers.14.self_attn.q_proj` | `(4096, 4096)` | 8192 | 310.3702 | 8192 | 310.3702 |
| `model.layers.14.self_attn.k_proj` | `(1024, 4096)` | 8192 | 316.9750 | 8192 | 316.9750 |
| `model.layers.14.self_attn.v_proj` | `(1024, 4096)` | 8192 | 373.2399 | 8192 | 373.2399 |
| `model.layers.14.self_attn.o_proj` | `(4096, 4096)` | 8192 | 342.4181 | 8192 | 342.4181 |
| `model.layers.14.mlp.gate_proj` | `(14336, 4096)` | 8192 | 320.2517 | 8192 | 320.2517 |
| `model.layers.14.mlp.up_proj` | `(14336, 4096)` | 8192 | 321.8136 | 8192 | 321.8136 |
| `model.layers.14.mlp.down_proj` | `(4096, 14336)` | 8192 | 363.5491 | 8192 | 363.5491 |
| `model.layers.15.self_attn.q_proj` | `(4096, 4096)` | 8192 | 314.1860 | 8192 | 314.1860 |
| `model.layers.15.self_attn.k_proj` | `(1024, 4096)` | 8192 | 314.6753 | 8192 | 314.6753 |
| `model.layers.15.self_attn.v_proj` | `(1024, 4096)` | 8192 | 400.0866 | 8192 | 400.0866 |
| `model.layers.15.self_attn.o_proj` | `(4096, 4096)` | 8192 | 356.2232 | 8192 | 356.2232 |
| `model.layers.15.mlp.gate_proj` | `(14336, 4096)` | 8192 | 330.7723 | 8192 | 330.7723 |
| `model.layers.15.mlp.up_proj` | `(14336, 4096)` | 8192 | 325.2821 | 8192 | 325.2821 |
| `model.layers.15.mlp.down_proj` | `(4096, 14336)` | 8192 | 343.0709 | 8192 | 343.0709 |
| `model.layers.16.self_attn.q_proj` | `(4096, 4096)` | 8192 | 311.8786 | 8192 | 311.8786 |
| `model.layers.16.self_attn.k_proj` | `(1024, 4096)` | 8192 | 314.7818 | 8192 | 314.7818 |
| `model.layers.16.self_attn.v_proj` | `(1024, 4096)` | 8192 | 338.7047 | 8192 | 338.7047 |
| `model.layers.16.self_attn.o_proj` | `(4096, 4096)` | 8192 | 347.8226 | 8192 | 347.8226 |
| `model.layers.16.mlp.gate_proj` | `(14336, 4096)` | 8192 | 336.3187 | 8192 | 336.3187 |
| `model.layers.16.mlp.up_proj` | `(14336, 4096)` | 8192 | 314.6022 | 8192 | 314.6022 |
| `model.layers.16.mlp.down_proj` | `(4096, 14336)` | 8192 | 368.5835 | 8192 | 368.5835 |
| `model.layers.17.self_attn.q_proj` | `(4096, 4096)` | 8192 | 314.3424 | 8192 | 314.3424 |
| `model.layers.17.self_attn.k_proj` | `(1024, 4096)` | 8192 | 319.6807 | 8192 | 319.6807 |
| `model.layers.17.self_attn.v_proj` | `(1024, 4096)` | 8192 | 349.3538 | 8192 | 349.3538 |
| `model.layers.17.self_attn.o_proj` | `(4096, 4096)` | 8192 | 348.2813 | 8192 | 348.2813 |
| `model.layers.17.mlp.gate_proj` | `(14336, 4096)` | 8192 | 326.5928 | 8192 | 326.5928 |
| `model.layers.17.mlp.up_proj` | `(14336, 4096)` | 8192 | 324.8428 | 8192 | 324.8428 |
| `model.layers.17.mlp.down_proj` | `(4096, 14336)` | 8192 | 343.2702 | 8192 | 343.2702 |
| `model.layers.18.self_attn.q_proj` | `(4096, 4096)` | 8192 | 316.2140 | 8192 | 316.2140 |
| `model.layers.18.self_attn.k_proj` | `(1024, 4096)` | 8192 | 316.6085 | 8192 | 316.6085 |
| `model.layers.18.self_attn.v_proj` | `(1024, 4096)` | 8192 | 342.8209 | 8192 | 342.8209 |
| `model.layers.18.self_attn.o_proj` | `(4096, 4096)` | 8192 | 338.6973 | 8192 | 338.6973 |
| `model.layers.18.mlp.gate_proj` | `(14336, 4096)` | 8192 | 324.2102 | 8192 | 324.2102 |
| `model.layers.18.mlp.up_proj` | `(14336, 4096)` | 8192 | 323.4813 | 8192 | 323.4813 |
| `model.layers.18.mlp.down_proj` | `(4096, 14336)` | 8192 | 346.9024 | 8192 | 346.9024 |
| `model.layers.19.self_attn.q_proj` | `(4096, 4096)` | 8192 | 314.8598 | 8192 | 314.8598 |
| `model.layers.19.self_attn.k_proj` | `(1024, 4096)` | 8192 | 314.3897 | 8192 | 314.3897 |
| `model.layers.19.self_attn.v_proj` | `(1024, 4096)` | 8192 | 339.4429 | 8192 | 339.4429 |
| `model.layers.19.self_attn.o_proj` | `(4096, 4096)` | 8192 | 336.8599 | 8192 | 336.8599 |
| `model.layers.19.mlp.gate_proj` | `(14336, 4096)` | 8192 | 313.9066 | 8192 | 313.9066 |
| `model.layers.19.mlp.up_proj` | `(14336, 4096)` | 8192 | 323.2712 | 8192 | 323.2712 |
| `model.layers.19.mlp.down_proj` | `(4096, 14336)` | 8192 | 340.4979 | 8192 | 340.4979 |
| `model.layers.20.self_attn.q_proj` | `(4096, 4096)` | 8192 | 314.5818 | 8192 | 314.5818 |
| `model.layers.20.self_attn.k_proj` | `(1024, 4096)` | 8192 | 313.9581 | 8192 | 313.9581 |
| `model.layers.20.self_attn.v_proj` | `(1024, 4096)` | 8192 | 335.6040 | 8192 | 335.6040 |
| `model.layers.20.self_attn.o_proj` | `(4096, 4096)` | 8192 | 330.3868 | 8192 | 330.3868 |
| `model.layers.20.mlp.gate_proj` | `(14336, 4096)` | 8192 | 331.4390 | 8192 | 331.4390 |
| `model.layers.20.mlp.up_proj` | `(14336, 4096)` | 8192 | 322.4181 | 8192 | 322.4181 |
| `model.layers.20.mlp.down_proj` | `(4096, 14336)` | 8192 | 351.5521 | 8192 | 351.5521 |
| `model.layers.21.self_attn.q_proj` | `(4096, 4096)` | 8192 | 314.1204 | 8192 | 314.1204 |
| `model.layers.21.self_attn.k_proj` | `(1024, 4096)` | 8192 | 314.2747 | 8192 | 314.2747 |
| `model.layers.21.self_attn.v_proj` | `(1024, 4096)` | 8192 | 349.3407 | 8192 | 349.3407 |
| `model.layers.21.self_attn.o_proj` | `(4096, 4096)` | 8192 | 339.7783 | 8192 | 339.7783 |
| `model.layers.21.mlp.gate_proj` | `(14336, 4096)` | 8192 | 318.7106 | 8192 | 318.7106 |
| `model.layers.21.mlp.up_proj` | `(14336, 4096)` | 8192 | 318.1945 | 8192 | 318.1945 |
| `model.layers.21.mlp.down_proj` | `(4096, 14336)` | 8192 | 338.7282 | 8192 | 338.7282 |
| `model.layers.22.self_attn.q_proj` | `(4096, 4096)` | 8192 | 311.6155 | 8192 | 311.6155 |
| `model.layers.22.self_attn.k_proj` | `(1024, 4096)` | 8192 | 313.1237 | 8192 | 313.1237 |
| `model.layers.22.self_attn.v_proj` | `(1024, 4096)` | 8192 | 313.0033 | 8192 | 313.0033 |
| `model.layers.22.self_attn.o_proj` | `(4096, 4096)` | 8192 | 321.4512 | 8192 | 321.4512 |
| `model.layers.22.mlp.gate_proj` | `(14336, 4096)` | 8192 | 318.1651 | 8192 | 318.1651 |
| `model.layers.22.mlp.up_proj` | `(14336, 4096)` | 8192 | 324.7101 | 8192 | 324.7101 |
| `model.layers.22.mlp.down_proj` | `(4096, 14336)` | 8192 | 352.9224 | 8192 | 352.9224 |
| `model.layers.23.self_attn.q_proj` | `(4096, 4096)` | 8192 | 312.9730 | 8192 | 312.9730 |
| `model.layers.23.self_attn.k_proj` | `(1024, 4096)` | 8192 | 312.1875 | 8192 | 312.1875 |
| `model.layers.23.self_attn.v_proj` | `(1024, 4096)` | 8192 | 328.2809 | 8192 | 328.2809 |
| `model.layers.23.self_attn.o_proj` | `(4096, 4096)` | 8192 | 320.8591 | 8192 | 320.8591 |
| `model.layers.23.mlp.gate_proj` | `(14336, 4096)` | 8192 | 309.6038 | 8192 | 309.6038 |
| `model.layers.23.mlp.up_proj` | `(14336, 4096)` | 8192 | 304.8419 | 8192 | 304.8419 |
| `model.layers.23.mlp.down_proj` | `(4096, 14336)` | 8192 | 321.0719 | 8192 | 321.0719 |
| `model.layers.24.self_attn.q_proj` | `(4096, 4096)` | 8192 | 313.3277 | 8192 | 313.3277 |
| `model.layers.24.self_attn.k_proj` | `(1024, 4096)` | 8192 | 313.6618 | 8192 | 313.6618 |
| `model.layers.24.self_attn.v_proj` | `(1024, 4096)` | 8192 | 318.3073 | 8192 | 318.3073 |
| `model.layers.24.self_attn.o_proj` | `(4096, 4096)` | 8192 | 325.6475 | 8192 | 325.6475 |
| `model.layers.24.mlp.gate_proj` | `(14336, 4096)` | 8192 | 319.1357 | 8192 | 319.1357 |
| `model.layers.24.mlp.up_proj` | `(14336, 4096)` | 8192 | 319.0833 | 8192 | 319.0833 |
| `model.layers.24.mlp.down_proj` | `(4096, 14336)` | 8192 | 327.8884 | 8192 | 327.8884 |
| `model.layers.25.self_attn.q_proj` | `(4096, 4096)` | 8192 | 312.7742 | 8192 | 312.7742 |
| `model.layers.25.self_attn.k_proj` | `(1024, 4096)` | 8192 | 311.9422 | 8192 | 311.9422 |
| `model.layers.25.self_attn.v_proj` | `(1024, 4096)` | 8192 | 341.0506 | 8192 | 341.0506 |
| `model.layers.25.self_attn.o_proj` | `(4096, 4096)` | 8192 | 327.6587 | 8192 | 327.6587 |
| `model.layers.25.mlp.gate_proj` | `(14336, 4096)` | 8192 | 318.2362 | 8192 | 318.2362 |
| `model.layers.25.mlp.up_proj` | `(14336, 4096)` | 8192 | 313.1918 | 8192 | 313.1918 |
| `model.layers.25.mlp.down_proj` | `(4096, 14336)` | 8192 | 332.3494 | 8192 | 332.3494 |
| `model.layers.26.self_attn.q_proj` | `(4096, 4096)` | 8192 | 311.8159 | 8192 | 311.8159 |
| `model.layers.26.self_attn.k_proj` | `(1024, 4096)` | 8192 | 313.2718 | 8192 | 313.2718 |
| `model.layers.26.self_attn.v_proj` | `(1024, 4096)` | 8192 | 357.0975 | 8192 | 357.0975 |
| `model.layers.26.self_attn.o_proj` | `(4096, 4096)` | 8192 | 338.7629 | 8192 | 338.7629 |
| `model.layers.26.mlp.gate_proj` | `(14336, 4096)` | 8192 | 327.4417 | 8192 | 327.4417 |
| `model.layers.26.mlp.up_proj` | `(14336, 4096)` | 8192 | 314.7103 | 8192 | 314.7103 |
| `model.layers.26.mlp.down_proj` | `(4096, 14336)` | 8192 | 324.6607 | 8192 | 324.6607 |
| `model.layers.27.self_attn.q_proj` | `(4096, 4096)` | 8192 | 309.4084 | 8192 | 309.4084 |
| `model.layers.27.self_attn.k_proj` | `(1024, 4096)` | 8192 | 310.7245 | 8192 | 310.7245 |
| `model.layers.27.self_attn.v_proj` | `(1024, 4096)` | 8192 | 323.8905 | 8192 | 323.8905 |
| `model.layers.27.self_attn.o_proj` | `(4096, 4096)` | 8192 | 318.3723 | 8192 | 318.3723 |
| `model.layers.27.mlp.gate_proj` | `(14336, 4096)` | 8192 | 318.1355 | 8192 | 318.1355 |
| `model.layers.27.mlp.up_proj` | `(14336, 4096)` | 8192 | 314.3954 | 8192 | 314.3954 |
| `model.layers.27.mlp.down_proj` | `(4096, 14336)` | 8192 | 335.1273 | 8192 | 335.1273 |
| `model.layers.28.self_attn.q_proj` | `(4096, 4096)` | 8192 | 313.2295 | 8192 | 313.2295 |
| `model.layers.28.self_attn.k_proj` | `(1024, 4096)` | 8192 | 313.5040 | 8192 | 313.5040 |
| `model.layers.28.self_attn.v_proj` | `(1024, 4096)` | 8192 | 343.4230 | 8192 | 343.4230 |
| `model.layers.28.self_attn.o_proj` | `(4096, 4096)` | 8192 | 330.3073 | 8192 | 330.3073 |
| `model.layers.28.mlp.gate_proj` | `(14336, 4096)` | 8192 | 331.4600 | 8192 | 331.4600 |
| `model.layers.28.mlp.up_proj` | `(14336, 4096)` | 8192 | 322.8607 | 8192 | 322.8607 |
| `model.layers.28.mlp.down_proj` | `(4096, 14336)` | 8192 | 334.1450 | 8192 | 334.1450 |
| `model.layers.29.self_attn.q_proj` | `(4096, 4096)` | 8192 | 311.2898 | 8192 | 311.2898 |
| `model.layers.29.self_attn.k_proj` | `(1024, 4096)` | 8192 | 314.0581 | 8192 | 314.0581 |
| `model.layers.29.self_attn.v_proj` | `(1024, 4096)` | 8192 | 327.4563 | 8192 | 327.4563 |
| `model.layers.29.self_attn.o_proj` | `(4096, 4096)` | 8192 | 330.4463 | 8192 | 330.4463 |
| `model.layers.29.mlp.gate_proj` | `(14336, 4096)` | 8192 | 330.9421 | 8192 | 330.9421 |
| `model.layers.29.mlp.up_proj` | `(14336, 4096)` | 8192 | 311.7183 | 8192 | 311.7183 |
| `model.layers.29.mlp.down_proj` | `(4096, 14336)` | 8192 | 332.8676 | 8192 | 332.8676 |
| `model.layers.30.self_attn.q_proj` | `(4096, 4096)` | 8192 | 314.3447 | 8192 | 314.3447 |
| `model.layers.30.self_attn.k_proj` | `(1024, 4096)` | 8192 | 315.6959 | 8192 | 315.6959 |
| `model.layers.30.self_attn.v_proj` | `(1024, 4096)` | 8192 | 342.0208 | 8192 | 342.0208 |
| `model.layers.30.self_attn.o_proj` | `(4096, 4096)` | 8192 | 325.9998 | 8192 | 325.9998 |
| `model.layers.30.mlp.gate_proj` | `(14336, 4096)` | 8192 | 313.2512 | 8192 | 313.2512 |
| `model.layers.30.mlp.up_proj` | `(14336, 4096)` | 8192 | 317.8566 | 8192 | 317.8566 |
| `model.layers.30.mlp.down_proj` | `(4096, 14336)` | 8192 | 343.6867 | 8192 | 343.6867 |
| `model.layers.31.self_attn.q_proj` | `(4096, 4096)` | 8192 | 313.0353 | 8192 | 313.0353 |
| `model.layers.31.self_attn.k_proj` | `(1024, 4096)` | 8192 | 314.3104 | 8192 | 314.3104 |
| `model.layers.31.self_attn.v_proj` | `(1024, 4096)` | 8192 | 325.4233 | 8192 | 325.4233 |
| `model.layers.31.self_attn.o_proj` | `(4096, 4096)` | 8192 | 335.7356 | 8192 | 335.7356 |
| `model.layers.31.mlp.gate_proj` | `(14336, 4096)` | 8192 | 323.8217 | 8192 | 323.8217 |
| `model.layers.31.mlp.up_proj` | `(14336, 4096)` | 8192 | 323.7858 | 8192 | 323.7858 |
| `model.layers.31.mlp.down_proj` | `(4096, 14336)` | 8192 | 407.9331 | 8192 | 407.9331 |
