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
For Metal runtime validation, use
[`docs/benchmarks/METAL_VALIDATION.md`](benchmarks/METAL_VALIDATION.md) and
preserve the emitted `metal_validation.json`/Markdown/log artifact set.

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
  contradictory forced modes fail loudly, and `mfma` selects the public
  F16-input/F32-output wrapper on HIP/gfx942 when the runtime transformer
  registry is present. The current Float32 ABI returns
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
  and strided-batch correctness coverage. Auto dispatch now keeps CTA128 for
  aligned large full-tile calls and defaults tail/general semantic calls to
  CTA64_EDGE until CTA128_EDGE has separate performance evidence. The ROCm
  benchmark harness now has opt-in wrapper rows for the headline MFMA path plus
  explicit alpha/beta, transpose, and strided-batched semantic checks against
  rocBLAS. The
  first packed-parameter wrapper fixture on MI300X / ROCm 7.2.4 measured
  `0.051632 ms` for 512³ (`1.967x` vs rocBLAS) and `0.033700 ms` for the edge
  shape 513x511x257 (`0.859x` vs rocBLAS), confirming correctness and removing
  the earlier per-scalar temporary-buffer overhead. A focused CTA128 evidence
  rerun with the public wrapper enabled passed correctness for 128³, 512³,
  1024³, 1024x512x1024, 512x1024x1024, and 2048x512x1024. That fixture also
  shows the remaining wrapper-level cost clearly: large tile-divisible wrapper
  rows are still slower than the raw MFMA timing rows because the public path
  includes runtime dispatch, launch setup, packed-parameter lifetime handling,
  and final synchronization. The next production work is continued
  shape-threshold tuning, cached/by-value launch-parameter handling to reduce
  wrapper overhead, and backend-specific expansion beyond HIP/gfx942.
- **Public-wrapper semantic validation and current ROCm evidence:** the
  `20260624_mi300x_wrapper_semantics` fixture passed the focused public ABI
  sweep on MI300X / ROCm 7.2.4: tail-small `0.023151 ms` (`0.534x` versus
  rocBLAS), tail-large `0.185373 ms` (`1.990x`), alpha/beta `0.038582 ms`
  (`1.164x`), transpose `0.025538 ms` (`1.251x`), and strided batch
  `0.018260 ms` (`1.041x` against `rocblas_gemm_strided_batched_ex`). The
  follow-up `20260624_mi300x_wrapper_semantic_fulltile` and
  `20260624_mi300x_wrapper_semantic_fulltile_50iter` fixtures validate the
  full-tile semantic route for tile-divisible alpha/beta and transpose calls.
  In the stable 50-iteration run, wrapper alpha/beta improved to
  `0.025392 ms` (`0.772x` versus rocBLAS), while wrapper transpose improved to
  `0.023297 ms` but remains performance-open at `1.153x`. The
  broader `20260624_mi300x_broad_wrapper_mfma_hipsparselt` fixture passed
  AXPY, F32 GEMM, portable F16/F32-output GEMM, raw MFMA variants, public
  F16/F32 wrapper rows, CSR SpMV, and hipSPARSELt structured GEMM. Key current
  production-candidate rows: CTA64 shared at 512³ `0.020573 ms` (`0.800x`
  versus rocBLAS), CTA128 at 1024³ `0.053449 ms` (`1.255x`), wrapper MFMA at
  1024³ `0.057410 ms` (`1.247x`), and hipSPARSELt structured GEMM at 512³
  `0.011817 ms` (`0.612x` versus dense pruned-A rocBLAS reference). The
  profile artifact confirms CTA128 remains scratch-free (`Scratch_Size=0`,
  `VGPR_Count=84`) and CTA128_EDGE remains the main performance gap
  (`0.180184 ms` at 1024³ in the profile matrix, about `4.005x` versus
  rocBLAS). The follow-up `20260624_mi300x_edge_tails_profile` fixture confirms
  the same behavior on true non-divisible tail shapes: CTA64_EDGE and
  CTA128_EDGE are both scratch-free and correctness-clean, but direct generated
  EDGE is still slower than rocBLAS. Wrapper auto dispatch chooses the vendor
  path for those true-tail shapes and stays at `0.999x` to `1.004x` versus
  rocBLAS. The `20260624_mi300x_edge_tails_store_fastpath_rocm724` fixture
  validates the alpha/beta store fast path on ROCm 7.2.4 and shows only
  noise-level direct EDGE improvement (`0.2%` to `0.6%`). The next GEMM work is
  therefore not basic correctness or simple epilogue branch cleanup. The
  `20260625_mi300x_edge_tails_predicate_hoist_rocm724` fixture validates the
  first CTA64_EDGE predicate-hoist cleanup. Correctness and the scratch-free
  resource envelope are preserved (`Scratch_Size=0`, `VGPR_Count=40`,
  `SGPR_Count=48` for CTA64_EDGE), but direct CTA64_EDGE timings changed only
  by about `+1%` on the sampled true-tail shapes. Predicate hoisting is useful
  as a source-shape cleanup, not as a throughput fix. The next EDGE performance
  work should move to fused true-tail kernels or a different EDGE kernel
  strategy, plus transpose-layout tuning. The
  `20260625_mi300x_edge_split_rocm724` fixture tested the first opt-in
  split/interior-border wrapper path and found it correctness-clean but slower
  than single-pass EDGE on all sampled true-tail shapes. Public AUTO therefore
  remains vendor-routed for these tails; split EDGE is diagnostic-only. The
  `20260625_mi300x_edge_nn_rocm724` fixture then validates a fused CTA64
  NN-only true-tail kernel that removes transpose, batching, and alpha/beta
  epilogue handling from the generic EDGE path. It is correctness-clean,
  scratch-free, and reduces register pressure (`VGPR_Count=24`,
  `SGPR_Count=32`) versus generic CTA64_EDGE (`VGPR_Count=40`,
  `SGPR_Count=48`). The timing gain is too small to change policy:
  direct NN fast-tail rows improve generic CTA64_EDGE by only about `2%` to
  `4%` and remain `1.165x` to `1.930x` versus rocBLAS. Public AUTO therefore
  continues to select vendor dispatch for the sampled true-tail shapes.

  A follow-up ROCm code-object inspection on the same host confirms the source
  of the EDGE cost. CTA64 shared full-tile disassembles to about `269`
  instructions, while CTA64_EDGE is about `878`; CTA128 full-tile is about
  `893`, while CTA128_EDGE is about `3580`. MFMA instruction count is unchanged
  (`1` for CTA64 and `16` for CTA128). The bloat is branch/address/control:
  CTA128_EDGE has about `494` branch/compare instructions, `1711` integer
  address/control ops, and `322` wait-count instructions versus `6`, `406`, and
  `50` respectively for CTA128 full-tile. This points toward specialized
  boundary CTAs, split interior/border launches, or fewer per-lane predicates as
  the next EDGE optimization family.

Implementation details and acceptance gates are tracked in the source
repository design notes. Public release reports should cite the generated JSON
and Markdown benchmark artifacts instead of internal issue trackers.

## Deferred Or Hardware-Specific Work

The following remain separate tracks:

- CUDA comparison against cuBLAS and cuSPARSE on an NVIDIA host.
- Vulkan/OpenCL tuning beyond correctness smoke checks.
- Metal tuning on Apple GPU hardware. The runtime now has diagnostic counters
  for command buffers, encoders, host-visible copies, and queue syncs, plus an
  experimental opt-in submit-batching path controlled by
  `NAVATALA_GPU_METAL_BATCH_SUBMITS` and
  `NAVATALA_GPU_METAL_BATCH_LIMIT`. It also has an experimental private
  device-buffer mode controlled by
  `NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS`, now backed by runtime
  offset-copy support. The validation runner, JSON schema, report renderer, and
  manual self-hosted macOS workflow are present; these runtime modes remain
  default-off until the Apple Silicon validation pass in
  [`docs/benchmarks/METAL_VALIDATION.md`](benchmarks/METAL_VALIDATION.md)
  records correctness and wall-time evidence.
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
