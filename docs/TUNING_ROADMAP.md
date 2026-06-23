# Backend Tuning Roadmap

Navatala GPU ships a broad portable kernel corpus first. The alpha release
does not claim that every generated kernel is hand-tuned for each backend. This
roadmap defines where backend-specific tuning should be applied and how tuning
evidence must be recorded.

## Release Position

- **Portable baseline:** current generated kernels are the default path unless
  `kernels/manifest.json` explicitly marks a backend implementation as
  vendor-backed or tuned.
- **No blanket performance claim:** source availability is not performance
  parity with vendor libraries.
- **Selective tuning:** tune only kernel families that dominate real workloads
  or unblock public benchmark credibility.

## Priority Families

| Priority | Family | Representative kernels | Why it matters | Benchmark target |
| --- | --- | --- | --- | --- |
| P0 | Sparse SpMV / solver vector ops | `navatala_sparse_csr_spmv_row_f32`, `navatala_sparse_axpy_f32`, dot/reduction kernels | Drives AMG/CG/PCG, CFD pressure solves, graph analytics | Compare against rocSPARSE/cuSPARSE SpMV and CPU reference on small/medium/large CSR matrices. |
| P0 | GEMM-like dense kernels | transformer tiled GEMM kernels, future public GEMM wrappers | ML and neural-operator workloads are sensitive to GEMM throughput | Compare against rocBLAS/cuBLAS for representative matrix shapes. [1] |
| P1 | Scan/sort primitives | dataframe radix/argsort/scan kernels | Shared by dataframe, sparse preprocessing, graph algorithms | Compare against CUB/rocPRIM where available; report throughput and correctness. |
| P1 | AMG primitives | restriction/prolongation, smoother kernels, residual kernels | Needed for OpenFOAM GPU pressure/momentum performance | Compare against current portable baseline in OpenFOAM GPU pressure benchmarks. |
| P1 | CFD finite-volume kernels | pressure correction, momentum predictor, VOF alpha update | Directly impacts marine CFD demos | Compare wall time inside OpenFOAM GPU benchmark cases. |
| P2 | DataFrame/groupby/join kernels | group-by reductions, hash joins, string kernels | Useful for analytics, but workload-dependent | Compare against cuDF/Polars CPU fixtures where practical. |

[1] MFMA prototype work landed in Option A. The broad ROCm benchmark now emits a
`GEMM_F16_PORTABLE_F32OUT` denominator row and the K-loop, CTA64 direct,
CTA64 shared-memory, CTA64 pipelined R6 staged-panel, and CTA128 MFMA variants,
so same-host reports can compare pressure-reduction and throughput trade-offs.
Production arbitrary-shape GEMM replacement work remains pending.

## Evidence Format

Each tuned family must add a report under `docs/benchmarks/` or attach a
release artifact with:

- public kernel name and backend;
- device model, driver/runtime versions, compiler flags;
- input shapes and dataset provenance;
- baseline implementation;
- tuned implementation identifier;
- correctness tolerance and reference result;
- wall time, throughput, memory bandwidth where meaningful;
- speedup/geomean summary and outliers;
- whether the tuned path is enabled by default.

For HIP/ROCm reports, use
[`docs/benchmarks/ROCM_VALIDATION_TEMPLATE.md`](benchmarks/ROCM_VALIDATION_TEMPLATE.md).
The opt-in ROCm vendor benchmark harness is documented in
[`docs/benchmarks/ROCM_VENDOR_BENCHMARKS.md`](benchmarks/ROCM_VENDOR_BENCHMARKS.md).

## Manifest Requirements

When a tuned or vendor-backed path is added, update the public manifest entry:

- raw kernel files remain `implementation.kind = portable_kernel` unless that
  specific backend source is itself tuned;
- wrapper/orchestrator vendor calls are recorded under
  `libraryOperations[].implementation.kind = vendor_dispatch`;
- `implementation.vendorBacked`: `true` only for actual vendor-library
  dispatch;
- `implementation.notes`: include the tuning/report reference;
- per-backend entries must not imply tuning unless that backend has evidence.

## First Tuning Candidates

1. `navatala_sparse_csr_spmv_row_f32` and `navatala_sparse_csr_spmv_row_f64`.
2. `navatala_sparse_axpy_f32` plus reduction/dot helpers used by solvers.
3. Transformer tiled GEMM Float32/Float16 kernels.
4. CFD pressure-solve SpMV/dot/AXPY path from the OpenFOAM GPU benchmark.

## Active P0 Work

The 2026-06-19 MI300X ROCm run produced enough hardware evidence to move two
P0 items into active tuning. Updated 2026-06-22 with the ROCm 7.2.4 follow-up run capturing acceptance evidence:

- **Adaptive CSR SpMV:** the generated HIP SpMV kernel is competitive for very
  short rows, but is 14.87x slower than rocSPARSE at `rows=262144,
  rowNnz=27`. The near-term fix is a subgroup/wave-reduction path for
  medium/high row-nnz while preserving the existing thread-per-row path for low
  row-nnz. Hardware acceptance gate met on 06-22 (rowNnz=27 at 2.80x vs <=3x target); real-CFD pressure-matrix benchmark remains the closure gate.
- **GEMM dispatch policy:** public GEMM wrappers should prefer rocBLAS/cuBLAS
  for supported CUDA/HIP large-shape calls. Generated tiled GEMM kernels remain
  portable fallback/reference paths unless their callers route through
  `LibraryOp.BLAS_GEMM`. The first public C ABI path,
  `navatala_gpu_gemm_f32`, supports this policy in the real runtime FFI through
  `NAVATALA_GPU_GEMM_VENDOR_MODE` and
  `NAVATALA_GPU_GEMM_VENDOR_THRESHOLD_FLOPS`. The ROCm benchmark harness has an
  opt-in `GEMM_F32_WRAPPER_VENDOR` row for same-host wrapper evidence. Wrapper-vendor parity captured 06-22 on MI300X at 128^3/512^3/1024^3 (ratio 1.000x); see the dated fixture. CUDA/cuBLAS and non-HIP backend comparisons remain pending. Accepted mode aliases are
  `auto`/`1`/`true`, `vendor`/`always`/`force`, and
  `portable`/`reference`/`fallback`/`0`/`false`.
  `NAVATALA_GPU_GEMM_IMPL=auto|vendor|mfma|portable` has higher precedence for
  wrapper-level validation runs: `vendor` and `portable` force those paths,
  contradictory forced modes fail loudly, and `mfma` is reserved for the future
  public F16/F32 wrapper. The current Float32 ABI returns
  `NAVATALA_NOT_IMPLEMENTED` for `GEMM_IMPL=mfma` instead of silently measuring
  the wrong implementation.
- **Portable GEMM fallback tuning:** scalar/register/tile cleanup is useful for
  fallback paths, but it is not expected to close the full MFMA/WMMA gap. The
  generated transformer `tiled_gemm_f16/f32` kernels now use real
  shared/threadgroup tiles instead of scalar global-load dot products. Hardware evidence captured 06-22: portable F32 GEMM at 6.95x at 1024^3 (down from pre-Option-C 21.36x). Labelling as tuned_kernel deferred until either the F16 path is re-measured post-Option-C or the Option A MFMA prototype lands a production kernel.
- **MFMA speedup denominator and pressure evidence:** the 06-23 CTA follow-up
  fixture captures `GEMM_F16_PORTABLE_F32OUT`, `GEMM_F16_MFMA`,
  `GEMM_F16_MFMA_CTA64_DIRECT`, `GEMM_F16_MFMA_CTA64_SHARED`, and
  `GEMM_F16_MFMA_CTA128` in same-host MI300X / ROCm 7.2.4 reports. CTA64 direct
  is the low-pressure diagnostic prototype: 64x64 CTA, direct global MFMA loads,
  one accumulator per wave, no shared-memory staging. The CTA64 shared-memory
  variant preserves that register envelope while staging a 2 KB LDS panel per
  K step. rocprof confirms CTA64 shared compiles with `Scratch_Size=0`,
  `VGPR_Count=40`, and `LDS_Block_Size=2048`, versus CTA128 at
  `Scratch_Size=12`, `VGPR_Count=128`, and 16 KB LDS. At 1024^3, CTA64 shared
  improves CTA64 direct from `0.108166 ms` to `0.059890 ms` and beats the
  portable F16/F32-output denominator (`0.179902 ms`) by about `3.00x`. It
  remains about `1.399x` slower than rocBLAS F16/F32 (`0.042812 ms`).
  Additional same-day diagnostics showed that a CTA64 early-barrier schedule was
  neutral (`0.060080 ms` at 1024^3) and padded LDS strides were slower
  (`0.073456 ms` at 1024^3, with higher VGPR/LDS pressure). The R6
  `GEMM_F16_MFMA_CTA64_PIPELINED` row was measured on the same MI300X / ROCm
  7.2.4 host with two-slot staged panels, vector-copy requests, and exact
  dynamic copy-group matching. The first R6 run was correct and scratch-free,
  but lowered the copy groups synchronously: at 1024^3 it measured
  `0.084913 ms`, slower than CTA64 shared (`0.059997 ms`) and about `1.983x`
  slower than rocBLAS F16/F32 (`0.042828 ms`). Resource metadata recorded
  `Scratch_Size=0`, `VGPR_Count=51`, `SGPR_Count=70`, and 4 KB LDS.
  The direct global-to-LDS follow-up lowerer then mapped aligned
  `CopyGlobalToShared` requests to `buffer_load_dword ... lds` and
  `AwaitCopyGroup` to `s_waitcnt vmcnt(0)` plus the synchronization barrier.
  That path remains correct and scratch-free, and improved the 1024^3
  pipelined timing to `0.060183 ms`, roughly tying CTA64 shared at
  `0.059940 ms`. It did not become the production GEMM path: at 512^3 it
  measured `0.027703 ms` versus CTA64 shared at `0.020553 ms`, and resource
  metadata rose to `VGPR_Count=73`, `SGPR_Count=94`, and 4 KB LDS. The next
  GEMM work is therefore direct-LDS register-pressure reduction and scheduling
  cleanup while preserving `Scratch_Size=0`, not simply admitting more async
  copy syntax.
- **CTA128 production-candidate evidence:** the
  `20260623_mi300x_cta128_evidence` fixture reprofiled CTA128 on the final
  helper-signature / launch-bounds generated code. Correctness passed across
  square and rectangular tile-divisible shapes, and rocprof reported
  `Scratch_Size=0`, `LDS_Block_Size=16384`, `VGPR_Count=84`,
  `Accum_VGPR_Count=68`, and `SGPR_Count=112`. CTA128 is now the large-shape
  candidate: at 1024^3 it measured `0.053518 ms` versus CTA64 shared at
  `0.060056 ms` and rocBLAS at `0.042751 ms`. CTA64 shared remains the
  medium-shape candidate: at 512^3 it measured `0.020572 ms` versus CTA128 at
  `0.026223 ms`. The public F16-input/F32-output wrapper now implements
  shape-aware HIP/gfx942 MFMA dispatch with edge-tile, alpha/beta, transpose,
  and strided-batch correctness coverage. The next production work is timing
  coverage for the edge-capable wrapper path, continued shape-threshold tuning,
  and any backend-specific expansion beyond HIP/gfx942.

Implementation details and acceptance gates are tracked in the source
repository design notes. Public release reports should cite the generated JSON
and Markdown benchmark artifacts instead of internal issue trackers.

## Deferred Or Hardware-Specific Work

The following remain separate tracks:

- CUDA comparison against cuBLAS and cuSPARSE on an NVIDIA host.
- Vulkan/OpenCL tuning beyond correctness smoke checks.
- Metal tuning on Apple GPU hardware.
- Full matrix-intrinsic IR support for MFMA/WMMA/TensorCore/cooperative-matrix
  operations. The Option A experimental prototype has landed and this track is
  now actively driven by `MATRIX_INTRINSIC_IR_R3.md`.

## Acceptance Gate For Beta

Before promoting beyond developer preview:

- at least one P0 family must have backend-specific tuned implementation
  evidence;
- HIP/ROCm must have correctness and benchmark evidence for the selected P0
  family if ROCm performance is marketed;
- docs must keep portable baseline, tuned kernel, and vendor-library dispatch
  labels separate.
