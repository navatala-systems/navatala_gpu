# Backends

Each GPU backend is independently enabled at configure time and gracefully
disabled when its toolkit is not present.

| Feature           | CUDA | HIP | Vulkan | OpenCL | Metal |
|-------------------|:----:|:---:|:------:|:------:|:-----:|
| Device memory     | yes  | yes | yes    | yes    | yes   |
| Pinned host mem.  | yes  | yes | yes    | yes    | yes   |
| Managed memory    | yes  | yes | no     | no     | yes   |
| Events            | yes  | yes | yes    | yes    | yes   |
| Graph capture     | yes  | yes | no     | no     | no    |
| Float64           | yes  | yes | yes    | yes    | no    |
| Float16           | yes  | yes | yes    | yes    | yes   |
| Subgroup / warp   | yes  | yes | yes    | yes    | yes   |
| JIT (text → PTX)  | yes  | yes | n/a    | yes    | yes   |
| AOT (binary)      | yes  | yes | yes    | no     | yes   |

## CUDA

Requires CUDA Toolkit 11.0 or newer with `nvcc`, NVRTC, and the CUDA driver
present. The runtime uses NVRTC to JIT-compile kernel sources to PTX, then
loads them via the CUDA driver API. Graph capture uses `cudaGraph_t` directly.

Set `GPU_RUNTIME_BACKEND=cuda` to force CUDA selection.

## HIP

Requires ROCm 5.0 or newer with `hipcc` and `hiprtc`. HIP is the AMD
equivalent of CUDA; almost all kernel source is shared between the two
backends (kernels live in separate `kernels/cuda/` and `kernels/hip/` trees
for clarity, but the bodies are nearly identical).

Set `GPU_RUNTIME_BACKEND=hip` to force HIP selection.

## Vulkan compute

Requires the Vulkan SDK with `glslc` available in `PATH`. The runtime
invokes `glslc` as a subprocess to compile GLSL compute shaders to SPIR-V,
caches the result, and submits via `vkCmdDispatch`.

Vulkan compute has no graph capture and no managed memory; pinned (host
visible) memory is the main lever for low-latency transfers.

Set `GPU_RUNTIME_BACKEND=vulkan` to force Vulkan selection.

## OpenCL

Requires OpenCL 1.2 or newer headers plus the system's ICD loader (typically
`libOpenCL.so` on Linux). The kernels are GLSL-flavoured OpenCL C; they
compile through the vendor's runtime compiler. OpenCL is the broadest
hardware coverage but tends to lag the other backends in feature parity.

Set `GPU_RUNTIME_BACKEND=opencl` to force OpenCL selection.

## Metal

macOS / iOS only. Requires Xcode Command Line Tools. Kernels are Metal
Shading Language. Note that Metal lacks first-class double-precision support
on most Apple silicon; double-precision kernels will not run on Apple GPUs
even when the source is present.

The Metal backend is implemented in Objective-C++ (`backend_metal.mm`) and
is built only on Apple platforms.

### Metal runtime tuning

The runtime includes diagnostic counters for Metal host overhead and an
experimental opt-in compute-submit batching mode. Enable counters with
`NAVATALA_GPU_RUNTIME_PROFILE=1`. Enable batching with
`NAVATALA_GPU_METAL_BATCH_SUBMITS=1`; `NAVATALA_GPU_METAL_BATCH_LIMIT`
controls how many consecutive compute dispatches are encoded before an
automatic flush. Batching is flushed before memory copies, queue
synchronization, event record/wait calls, and queue destruction.

`NAVATALA_GPU_METAL_PRIVATE_DEVICE_BUFFERS=1` is an experimental mode that
allocates `MemoryKind::Device` buffers as Metal private storage. It relies on
queued staging copies and offset-capable blits for public C ABI H2D/D2H/D2D
operations.

Submit batching and private device buffers are default-off until validated on
Apple Silicon. See
[`benchmarks/METAL_VALIDATION.md`](benchmarks/METAL_VALIDATION.md) for the
required correctness and profiling pass.

The one-command validation path is:

```bash
./scripts/run_metal_validation.sh
```

It emits a `metal_validation.json` artifact that is checked by
`scripts/validate_metal_validation_json.py` and can be rendered with
`scripts/render_metal_validation_report.py`.

### Metal coverage caveat

Apple GPUs effectively have no double-precision support, so the Metal
corpus is much smaller than the other backends — about 60% the size of
the CUDA corpus. The omitted kernels are overwhelmingly the F64-only
ones; any algorithm that runs in single precision has a Metal
translation. See the per-backend kernel breakdown in
[`KERNELS.md`](KERNELS.md) for exact numbers and the full root-cause
analysis.

## Auto-selection order

When no `GPU_RUNTIME_BACKEND` is set, the runtime probes backends in this
order and returns the first that successfully initializes:

1. Metal (Apple platforms only)
2. HIP
3. CUDA
4. Vulkan
5. OpenCL

The intent is to prefer the most native backend per platform. Override the
auto-detection by setting the environment variable.

## Cross-backend kernel consistency

Kernels in `kernels/` are translated from a single upstream specification
into the supported backend forms where the backend can represent the
required features. Coverage is intentionally explicit rather than implied;
see [`BACKEND_COVERAGE.md`](BACKEND_COVERAGE.md) and
`../kernels/manifest.json` for the generated matrix.

Cross-backend numerical drift, when it occurs, is almost always traceable to:

- Single vs double precision (some backends transparently downcast).
- Hardware differences in rounding for transcendental functions
  (`exp`, `log`, `sqrt`, etc.).
- Subgroup / warp size differences (32 on NVIDIA/Vulkan, 64 on AMD CDNA,
  32 on AMD RDNA, varies on Metal/Intel).

If you find a real semantic divergence between backends for the same
available kernel, please file a bug — that's a spec defect, not an expected
difference.

## Portable Kernels And Vendor Libraries

The source corpus is primarily portable Navatala kernel code. A kernel being
present for CUDA or HIP means the operation has source available for that
backend; it does not mean the operation dispatches through cuBLAS, cuSPARSE,
cuDNN, rocBLAS, rocSPARSE, hipSPARSELt, or another vendor library.

Vendor-library integrations are useful for peak performance and should be
benchmarked separately when they are added. Until a path is explicitly
documented as vendor-backed, assume it is a portable implementation. Numerical
validation and performance evidence are tracked separately in
[`NUMERICAL_CONFORMANCE.md`](NUMERICAL_CONFORMANCE.md).

For HIP/ROCm, the public tree includes an opt-in benchmark harness that
compares selected generated kernels against rocBLAS and rocSPARSE. It can also
include a real hipSPARSELt structured sparse GEMM row for supported FP16/F32
compute shapes. See
[`benchmarks/ROCM_VENDOR_BENCHMARKS.md`](benchmarks/ROCM_VENDOR_BENCHMARKS.md).
