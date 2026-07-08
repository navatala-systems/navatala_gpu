# Metal Runtime Benchmark

- Schema: `navatala_gpu.metal_runtime_benchmark_matrix.v1`
- Generated: `2026-07-08T01:45:27Z`
- Commit: `navatala:3eee3d6;runtime:9cc277c`
- Host: `26.5.2` `arm64`; Chipset Model: Apple M4 Max; Vendor: Apple (0x106b); Metal Support: Metal 4
- Mode: `full`

## Runtime Counters

| Run | Flags | Pass | Command buffers | Compute encoders | Blit encoders | Host-visible copies | Batch dispatch | Batch flush | Archive hit/miss/store |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | baseline shared buffers | yes | 2178 | 2178 | 0 | 1594 | 0 | 0 | 0/1/1 |
| batch_submits | submit batching(limit=64) | yes | 132 | 132 | 0 | 1594 | 2178 | 132 | 1/0/0 |
| private | private buffers | yes | 3772 | 2178 | 1594 | 0 | 0 | 0 | 1/0/0 |
| private_batch_blits | private buffers, blit batching(limit=64) | yes | 2248 | 2178 | 70 | 0 | 0 | 0 | 1/0/0 |

## Timings

| Run | submit_sync mean ms | submit_burst mean ms | copy_roundtrip mean ms | copy_roundtrip GB/s |
|---|---:|---:|---:|---:|
| baseline | 0.268 | 1.095 | 1.332 | 75.6 |
| batch_submits | 0.140 | 1.032 | 1.342 | 75.0 |
| private | 0.155 | 1.096 | 0.368 | 273.4 |
| private_batch_blits | 0.117 | 1.087 | 0.307 | 328.2 |

## Interpretation

This benchmark is a runtime-overhead probe. It uses inline MSL and the public `gpu_runtime` API, so it can run on a Metal-only generated tree. It does not replace generated-kernel throughput fixtures.

Use the counter deltas to validate command-buffer and blit-encoder optimizations; use timing rows as host-local evidence only, because Apple Silicon scheduling and thermal state can shift absolute timings.
