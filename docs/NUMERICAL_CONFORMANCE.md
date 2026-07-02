# Numerical Conformance Status

This page tracks validation of Navatala GPU separately from source coverage.
`docs/BACKEND_COVERAGE.md` answers "does a backend source file exist for this
operation?". This page answers "has the operation been compiled and checked
against reference data on real hardware?".

## Current Release Position

Navatala GPU is a developer-preview release. The runtime, Python bindings, and
kernel corpus are packaged for inspection and early integration. Numerical
validation is intentionally conservative and backend-specific.

| Area | Status | Notes |
| --- | --- | --- |
| CPU-only packaging checks | Passed | Source tree hygiene, source distribution build, wheel metadata, import smoke tests, and manifest-driven Python tests run without requiring a GPU. |
| Manifest/file conformance | Passed | No-GPU CTest `manifest_conformance` validates manifest shape, advertised kernel source/binary paths, and representative family coverage. |
| CUDA runtime smoke | Passed | CUDA backend has been exercised on an RTX 4090 host with runtime `ctest`, direct square/SAXPY execution, and Python package import/manifest tests. |
| HIP/ROCm runtime smoke | Passed | `scripts/run_rocm_runtime_smoke.sh` passed on AMD Instinct MI300X VF / gfx942 with ROCm 7.2.4. |
| OpenCL runtime smoke | Partial | OpenCL source coverage is present. Device/driver availability varies by host, so release evidence is not yet treated as complete conformance. |
| Vulkan runtime smoke | Partial | Vulkan source and SPIR-V coverage are present. Full numerical comparison across the kernel corpus is pending. |
| Metal runtime smoke | Pending hardware | Metal sources are present where the backend can represent the operation. Runtime profile counters, opt-in submit batching, offset-copy support, opt-in private device buffers, and the `scripts/run_metal_validation.sh` artifact workflow are implemented, but Apple GPU correctness/performance validation is pending. |
| Cross-backend numerical comparisons | Pending | Broad per-operation tolerances and reference-vector reports are being expanded. |
| ROCm vendor benchmark harness | Passed on MI300X | Optional HIP benchmark target compares selected generated kernels against rocBLAS/rocSPARSE and can include an opt-in hipSPARSELt structured sparse GEMM row. Dated MI300X fixtures are checked in under `benchmarks/fixtures/hardware_runs/`, including the current `20260702_mi300x_broad_rocm724`, `20260702_mi300x_wrapper_semantics_rocm724`, and `20260702_mi300x_edge_tails_rocm724` refresh runs. |
| Vendor-library benchmark comparisons | Partial | ROCm AXPY/GEMM/CSR SpMV, public F16/F32 MFMA wrapper semantics, rocBLAS strided-batched GEMM, and opt-in hipSPARSELt structured GEMM evidence exists for MI300X. CUDA cuBLAS/cuSPARSE on NVIDIA host still pending. |

## What Has Been Validated

The current package has passed the following local gates before publication:

- Public tree hygiene: no private source paths, credentials, or internal
  generator namespace leaks in user-facing files.
- No-GPU runtime build with optional GPU backends disabled.
- Example build and CPU-only skip behaviour.
- Python source distribution and wheel build smoke checks.
- Python import, package metadata, ABI version, and manifest consistency checks.
- Public kernel manifest/file conformance through CTest.
- CUDA smoke execution on real NVIDIA hardware for representative runtime and
  Python-package paths.
- HIP/ROCm runtime smoke and ROCm vendor benchmark execution on AMD Instinct
  MI300X VF for representative rocBLAS, rocSPARSE, and hipSPARSELt rows.
- Public HIP/gfx942 F16-input/F32-output MFMA wrapper correctness for tail
  tiles, alpha/beta, transpose, and strided batching. The strided-batch row
  validates the `rocblas_gemm_strided_batched_ex` comparison path on MI300X.
- July 2026 MI300X / ROCm 7.2.4 refresh validation of the current public tree:
  runtime smoke passed, broad ROCm benchmark passed with `--require-full
  --require-hipsparselt`, and focused wrapper-semantics / true-tail matrices
  passed correctness.
- Dated MI300X broad ROCm benchmark fixture validation and Markdown rendering
  through the no-GPU CTest path.

These checks are sufficient for an alpha package, but they are not a complete
numerical certification of every backend/kernel pair.

## Reference Rules

Future conformance reports should state, for each operation family:

- The public operation name and dtype.
- The backend and device model.
- Input-shape ranges and random seeds.
- Reference implementation used for comparison.
- Absolute and relative tolerances.
- Determinism mode, if relevant.
- Whether the operation uses a portable Navatala kernel or a vendor-library
  dispatch path.

Use [`docs/benchmarks/ROCM_VALIDATION_TEMPLATE.md`](benchmarks/ROCM_VALIDATION_TEMPLATE.md)
for public HIP/ROCm validation and benchmark reports.
The optional ROCm benchmark harness is documented in
[`docs/benchmarks/ROCM_VENDOR_BENCHMARKS.md`](benchmarks/ROCM_VENDOR_BENCHMARKS.md).
Use [`docs/benchmarks/METAL_VALIDATION.md`](benchmarks/METAL_VALIDATION.md)
for the Apple Silicon Metal correctness and runtime-overhead validation pass.

Floating-point comparisons should use both absolute and relative tolerances.
Operations that rely on reductions, atomics, subgroup collectives, or
transcendental functions must publish backend-specific tolerances because
valid evaluation order and hardware rounding behaviour can differ.

## Portable Kernels Versus Vendor Libraries

Most operations in this release are portable kernel implementations emitted
for CUDA, HIP, OpenCL, Vulkan compute, and Metal where possible. They are meant
to provide a broad cross-vendor baseline and a stable runtime surface.

They are not advertised as tuned replacements for vendor libraries. In
particular:

- BLAS, sparse, solver, FFT, DNN, and graph-style operations may have portable
  kernels without a corresponding public `cublas*`, `rocblas*`, `cusparse*`,
  `rocsparse*`, `cudnn*`, or similar vendor symbol.
- Backend-specific vendor dispatch is limited to explicit wrapper paths where
  documented; for example, the real C ABI `navatala_gpu_gemm_f32` may route
  large CUDA/HIP Float32 GEMM calls through cuBLAS/rocBLAS when BLAS support and
  the configured dispatch policy allow it. These paths are described in
  `kernels/manifest.json` under `libraryOperations`, while standalone kernel
  source files remain marked as portable unless independently tuned.
- Performance claims require benchmark reports against the relevant vendor
  library on the same hardware.

The near-term GEMM dispatch policy is:

- public CUDA/HIP GEMM wrappers may prefer `LibraryOp.BLAS_GEMM` vendor dispatch
  when backend support, dtype/layout, and shape thresholds allow it;
- generated tiled GEMM kernels remain portable fallback/reference kernels unless
  their callers are explicitly routed through a vendor-backed wrapper;
- the portable transformer tiled GEMM kernels now use shared/threadgroup 16x16
  tiling, but they still require fresh same-host benchmark evidence before any
  tuned-kernel performance claim;
- `NAVATALA_GPU_GEMM_VENDOR_MODE` and
  `NAVATALA_GPU_GEMM_VENDOR_THRESHOLD_FLOPS` control the first public C ABI
  dispatch path; `auto` may fall back when dispatch is unavailable, while
  explicit `vendor`/`always` mode fails loudly if BLAS dispatch is unsupported;
  disabling vendor dispatch uses the available fallback/reference
  implementation, not an automatically tuned vendor-equivalent kernel; accepted
  mode aliases are `auto`/`1`/`true`, `vendor`/`always`/`force`, and
  `portable`/`reference`/`fallback`/`0`/`false`;
- `NAVATALA_GPU_GEMM_IMPL=auto|vendor|mfma|portable` is the higher-precedence
  wrapper implementation selector. `vendor` forces BLAS dispatch, `portable`
  forces the fallback/reference path, and contradictory combinations with
  `NAVATALA_GPU_GEMM_VENDOR_MODE` fail loudly. `mfma` selects the HIP/gfx942
  F16-input/F32-output MFMA wrapper when that backend registry shard is present;
  the Float32 ABI still returns `NAVATALA_NOT_IMPLEMENTED` for `mfma` rather
  than silently using an incompatible path;
- benchmark reports must label `portable_kernel`, `tuned_kernel`, and
  `vendor_library` paths separately.
- MFMA runtime dispatch is wired for the public F16-input/F32-output wrapper on
  HIP/gfx942. The wrapper uses edge-capable CTA64/CTA128 kernels with
  alpha/beta, transpose, and strided-batch correctness coverage. Auto dispatch
  defaults tail/general semantic calls to CTA64_EDGE; CTA128_EDGE remains a
  forced diagnostic route until edge-performance evidence improves. The
  CTA64/CTA128 benchmark rows remain prototype timing rows; the
  `20260623_mi300x_cta128_evidence` and
  `20260624_mi300x_cta128_edge_wrapper` fixtures record the shape-aware policy
  basis and wrapper validation evidence. The
  `20260624_mi300x_wrapper_mfma_packed_params_standard` fixture records the
  first direct timing rows for the public wrapper after scalar launch
  parameters were packed into one device buffer. The companion
  `20260624_mi300x_wrapper_mfma_cta128_evidence` fixture extends the same
  public-wrapper correctness evidence over the focused CTA128 shape matrix
  (128³, 512³, 1024³, and rectangular 1024/2048-class cases). Those rows all
  pass numerically, but the large-shape timings remain slower than the raw MFMA
  micro-benchmark rows because the public wrapper still includes runtime
  dispatch, launch setup, packed-parameter lifetime management, and a final
  synchronization. The `20260624_mi300x_wrapper_semantics` fixture records the
  first focused semantic sweep: tail-small `0.534x`, tail-large `1.990x`,
  alpha/beta `1.164x`, transpose `1.251x`, and strided-batch `1.041x` versus
  rocBLAS, all passing correctness. The
  `20260624_mi300x_wrapper_semantic_fulltile_50iter` fixture supersedes the
  alpha/beta and transpose performance rows for tile-divisible single-batch
  calls: wrapper alpha/beta is `0.772x` versus rocBLAS, while wrapper transpose
  is correctness-clean but still performance-open at `1.153x`. The
  `20260624_mi300x_broad_wrapper_mfma_hipsparselt` fixture records the broad
  sweep: the public wrapper passes aligned, tail, alpha/beta, transpose, and
  batched rows; CTA128 reaches `0.053449 ms` at 1024³ (`1.255x` versus
  rocBLAS); CTA64 shared reaches `0.020573 ms` at 512³ (`0.800x` versus
  rocBLAS); and hipSPARSELt structured GEMM passes, including `0.011817 ms`
  at 512³ (`0.612x` versus the dense pruned-A rocBLAS reference).
  The `20260624_mi300x_edge_tails_profile` fixture records true non-divisible
  tail shapes. All direct CTA64_EDGE, direct CTA128_EDGE, forced wrapper MFMA,
  wrapper auto, and wrapper vendor rows pass correctness. The resource profile
  confirms CTA64_EDGE and CTA128_EDGE are scratch-free, but direct generated
  EDGE remains slower than rocBLAS on the sampled true-tail shapes. Wrapper auto
  therefore selects the vendor path when available and stays at vendor parity
  (`0.999x` to `1.004x` in the fixture).
  The follow-up `20260624_mi300x_edge_tails_store_fastpath_rocm724` fixture
  validates the alpha/beta store fast path on ROCm 7.2.4. Correctness and
  scratch-free resource metadata are preserved, but direct EDGE timing changes
  are noise-level (`0.2%` to `0.6%` on sampled direct EDGE rows), so the
  remaining EDGE performance gap is not primarily epilogue alpha/beta branch
  overhead. The
  `20260625_mi300x_edge_tails_predicate_hoist_rocm724` fixture validates the
  subsequent CTA64_EDGE predicate-hoist cleanup. It also preserves correctness
  and the scratch-free resource envelope, but direct CTA64_EDGE timings remain
  effectively unchanged at about `+1%` versus the store-fastpath fixture. Public
  auto dispatch continues to select the vendor path for these sampled true-tail
  shapes. The `20260625_mi300x_edge_split_rocm724` fixture validates the
  opt-in split EDGE wrapper path. Split rows are correctness-clean, but slower
  than the existing single-pass forced MFMA EDGE path and much slower than
  vendor-routed AUTO, so split EDGE remains diagnostic-only. The follow-up
  `20260625_mi300x_edge_nn_rocm724` fixture validates a narrower CTA64 NN-only
  fast-tail candidate. It is limited to `NN`, `alpha = 1`, `beta = 0`, and
  single-batch true-tail calls. It is correctness-clean and scratch-free, and
  lowers resource pressure from `VGPR_Count=40`, `SGPR_Count=48` for generic
  CTA64_EDGE to `VGPR_Count=24`, `SGPR_Count=32`. Timing improves only modestly
  over generic CTA64_EDGE (`~2%` to `~4%` on sampled direct rows) and remains
  slower than rocBLAS, so public AUTO remains vendor-routed for these tails.
  The `20260702_mi300x_broad_rocm724` refresh confirms the same release
  position on the current public tree: CSR SpMV rowNnz=27 remains inside the
  `<= 3x` gate at `2.348x`, hipSPARSELt structured GEMM at 512³ remains
  vendor-beating at `0.585x`, alpha/beta remains faster than rocBLAS at
  `0.756x`, transpose remains the main Phase 2 generated-path gap at `1.145x`,
  and wrapper AUTO remains vendor-routed for large true-tail GEMM.

## Known Gaps Before Beta

- Public HIP/ROCm CI runner with correctness smoke tests.
- Real OpenFOAM pressure-matrix CSR distribution benchmark (the synthetic
  adaptive SpMV rowNnz=15/27 acceptance gates have been passed and captured in
  the MI300X fixtures; the 2026-07-02 broad run records rowNnz=27 at `2.348x`
  versus rocSPARSE).
- Full conformance-vector corpus for Python bindings and public C++ wrappers.
- Metal validation on macOS hardware, including before/after evidence for
  `NAVATALA_GPU_METAL_BATCH_SUBMITS` and correctness/performance evidence for
  `NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS`. The local schema, renderer, and
  sample fixture are in place; the missing item is a real Apple Silicon
  `metal_validation.json` fixture.
- Explicit tolerance files checked into the release tree for operation
  families where backend drift is expected.

## Reporting Issues

When reporting a numerical issue, include:

- Backend and device name.
- Operation name and dtype.
- Input dimensions and seed/data sample.
- Expected result source.
- Observed maximum absolute and relative error.
- Whether the result is nondeterministic across repeated runs.
