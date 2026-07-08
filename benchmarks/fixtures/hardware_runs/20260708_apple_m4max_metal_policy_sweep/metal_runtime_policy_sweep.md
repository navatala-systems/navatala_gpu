# Metal Runtime Policy Sweep

- Schema: `navatala_gpu.metal_runtime_policy_sweep.v1`
- Generated: `2026-07-08T01:45:43Z`
- Commit: `navatala:3eee3d6;runtime:9cc277c`
- Policy limit for size sweep: `64`
- Private-buffer minimum bytes: `0`

## Batch-Limit Sweep

| Limit | batch_submits command buffers | private blit encoders | submit_burst ms | private+blit copy GB/s |
|---:|---:|---:|---:|---:|
| 8 | 330 | 202 | 1.031 | 329.7 |
| 16 | 198 | 136 | 1.059 | 325.9 |
| 32 | 132 | 70 | 1.049 | 321.9 |
| 64 | 132 | 70 | 1.032 | 328.2 |
| 128 | 132 | 70 | 1.084 | 320.3 |

## Buffer-Size Sweep

| Elements | Bytes | baseline copy GB/s | private copy GB/s | private+blit copy GB/s | private+blit copy ms |
|---:|---:|---:|---:|---:|---:|
| 65536 | 262144 | 79.2 | 18.1 | 46.0 | 0.137 |
| 131072 | 524288 | 78.7 | 34.9 | 82.8 | 0.152 |
| 196608 | 786432 | 78.5 | 52.9 | 128.3 | 0.147 |
| 262144 | 1048576 | 79.5 | 67.2 | 165.2 | 0.152 |
| 524288 | 2097152 | 79.8 | 148.8 | 290.2 | 0.173 |
| 1048576 | 4194304 | 76.8 | 257.5 | 313.5 | 0.321 |
| 4194304 | 16777216 | 71.8 | 295.4 | 314.1 | 1.282 |

## Policy Reading

- Batch/blit limit `32` reaches the command-buffer and blit-encoder plateau for this benchmark; `64` is a conservative opt-in default.
- Shared buffers remain preferable for very small transfers; private buffers with batched blits become attractive around 512 KiB and strongly favorable from roughly 1 MiB upward.
- Use `NAVATALA_GPU_METAL_PRIVATE_MIN_BYTES` to keep small runtime buffers shared while allowing larger buffers to use private storage.
- These are local runtime-policy probes. Keep the knobs opt-in until the regenerated public tree has release-corpus benchmark fixtures.
