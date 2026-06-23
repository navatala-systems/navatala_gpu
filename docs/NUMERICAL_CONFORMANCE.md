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
| Metal runtime smoke | Pending hardware | Metal sources are present where the backend can represent the operation. Apple GPU validation is pending. |
| Cross-backend numerical comparisons | Pending | Broad per-operation tolerances and reference-vector reports are being expanded. |
| ROCm vendor benchmark harness | Passed on MI300X | Optional HIP benchmark target compares selected generated kernels against rocBLAS/rocSPARSE and can include an opt-in hipSPARSELt structured sparse GEMM row. A dated MI300X fixture is checked into `benchmarks/fixtures/rocm_vendor_benchmark_mi300x_20260622_rocm724.json` (the 06-19 fixture is retained as historical evidence). |
| Vendor-library benchmark comparisons | Partial | ROCm AXPY/GEMM/CSR SpMV and opt-in hipSPARSELt structured GEMM evidence exists for MI300X. Passed on MI300X for ROCm wrapper-vendor GEMM and hipSPARSELt structured GEMM; CUDA cuBLAS/cuSPARSE on NVIDIA host still pending. |

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
  alpha/beta, transpose, and strided-batch correctness coverage. The
  CTA64/CTA128 benchmark rows remain prototype timing rows; the
  `20260623_mi300x_cta128_evidence` and
  `20260624_mi300x_cta128_edge_wrapper` fixtures record the shape-aware policy
  basis and wrapper validation evidence. The
  `20260624_mi300x_wrapper_mfma_packed_params_standard` fixture records the
  first direct timing rows for the public wrapper after scalar launch
  parameters were packed into one device buffer.

## Known Gaps Before Beta

- Public HIP/ROCm CI runner with correctness smoke tests.
- Real OpenFOAM pressure-matrix CSR distribution benchmark (the synthetic
  adaptive SpMV rowNnz=15/27 acceptance gates have been passed and captured in
  the 06-22 fixture).
- Full conformance-vector corpus for Python bindings and public C++ wrappers.
- Metal validation on macOS hardware.
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
