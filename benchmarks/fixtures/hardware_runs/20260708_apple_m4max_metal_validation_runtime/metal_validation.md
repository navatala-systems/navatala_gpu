# Metal Validation Report

- Generated at: `2026-07-08T01:45:15Z`
- Source JSON: `metal_validation.json`
- Schema: `navatala_gpu.metal_validation.v1`
- Timing mode: `ctest_runtime_profile`
- Commit: `navatala:3eee3d6;runtime:9cc277c`
- Host: `Darwin 26.5.2 (arm64)`
- Xcode: `Xcode 26.6`
- Device summary: `Chipset Model: Apple M4 Max; Vendor: Apple (0x106b); Metal Support: Metal 4`
- Build directory: `/tmp/navatala_gpu_metal_fixture_tree/build-metal-validation`
- CMake: `cmake version 3.26.0-rc1`
- Compiler: `Apple clang version 21.0.0 (clang-2100.1.1.101)`
- Configuration: `Release`

## Runs

| Run | Batching | Blit batching | Private buffers | CTest | Wall s | submit | sync | command buffers | compute encoders | blit encoders | host-visible copies | batch dispatches | skipped empty syncs | archive hit/miss/store |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | false | false | false | pass | 1.070 | 2 | 256 | 9 | 2 | 2 | 19 | 0 | 250 | 0/2/2 |
| batched | true | false | false | pass | 0.170 | 2 | 256 | 9 | 2 | 2 | 19 | 2 | 250 | 2/0/0 |
| private | false | true | true | pass | 0.170 | 2 | 256 | 9 | 2 | 2 | 19 | 0 | 250 | 2/0/0 |

## Interpretation Notes

- This artifact is a runtime correctness and overhead probe, not a broad Metal performance benchmark.
- `command_buffer`, `compute_encoder`, and `blit_encoder` come from `NAVATALA_GPU_RUNTIME_PROFILE=1`.
- The batched run should reduce command-buffer pressure for command-buffer-bound workloads.
- Private-buffer mode is experimental until Apple hardware runs show correctness and acceptable copy overhead.
