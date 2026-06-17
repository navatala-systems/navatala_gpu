# Navatala GPU

Cross-platform GPU compute runtime and kernel corpus for scientific computing,
released under the Apache License 2.0.

This distribution bundles two cooperating layers:

1. **`runtime/`** — a C++20 abstraction that presents one API over CUDA, HIP,
   Vulkan compute, OpenCL, and Metal. Handles device enumeration, memory
   allocation (device, pinned, managed), execution queues, event-based
   synchronization, and CUDA/HIP graph capture.

2. **`kernels/`** — a corpus of compute kernels covering finite-volume CFD
   primitives, algebraic multigrid (AMG), classical iterative solvers
   (CG, BiCGSTAB, IDR, GMRES), sparse and dense BLAS, and a cross-platform
   machine-learning library (clustering, regression, KNN, decision trees,
   SVM, ARIMA, SHAP, UMAP, and more). Kernels ship in five backend forms
   (CUDA, HIP, OpenCL, Vulkan compute + SPIR-V, Metal) with consistent
   behaviour across vendors. Per-backend coverage is **not uniform** —
   see [`docs/BACKEND_COVERAGE.md`](docs/BACKEND_COVERAGE.md) for the
   current matrix.

   A host-side kernel registry that wraps the kernel files for runtime
   lookup ships under `runtime/include/navatala/` (header) and
   `runtime/src/internal/` (source). It ships as code but does not carry a
   CMakeLists.txt in this release.

3. **`orchestrator/`** — example host orchestrator code built on the
   runtime, demonstrating how the CFD kernels compose into a
   Volume-of-Fluid pressure-projection workflow
   (`Navatala::Cfd::VofPressureOrchestrator`). Worked example, not a
   production solver; ships as code without a turnkey CMakeLists.

## Status

This is a developer-preview / alpha release. The runtime library and kernel
corpus are both in active use for CFD workloads, but the public packaging,
documentation, CI matrix, and backend conformance reports are still being
expanded.

## Building

Prerequisites depend on the backends you enable.

| Backend | Required at build time                              |
| ------- | --------------------------------------------------- |
| CUDA    | CUDA Toolkit 11.0+ (`nvcc`, NVRTC, CUDA driver)     |
| HIP     | ROCm 5.0+ (`hipcc`, hipRTC)                         |
| Vulkan  | Vulkan SDK with `glslc` for GLSL→SPIR-V compilation |
| OpenCL  | OpenCL 1.2+ headers and ICD loader                  |
| Metal   | macOS 11+ with Xcode Command Line Tools             |

```bash
cmake -S . -B build
cmake --build build -j

# Run tests (requires at least one GPU backend to be available)
ctest --test-dir build --output-on-failure
```

Disable backends you don't need:

```bash
cmake -S . -B build \
    -DNAVATALA_GPU_USE_CUDA=OFF \
    -DNAVATALA_GPU_USE_HIP=ON \
    -DNAVATALA_GPU_USE_VULKAN=OFF \
    -DNAVATALA_GPU_USE_OPENCL=OFF
```

## Quick example

A complete, runnable example that loads one CUDA kernel from `kernels/cuda/`
and verifies the result is at [`examples/axpy_example.cpp`](examples/axpy_example.cpp).
After building, run it with:

```bash
./build/examples/axpy_example
```

The example exits 0 with a `[skip]` message on hosts without a GPU, so it
is safe to wire into CI even on CPU-only runners.

For a fuller tour, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how runtime and kernels fit together.
- [`docs/BACKENDS.md`](docs/BACKENDS.md) — per-backend capabilities and limitations.
- [`docs/KERNELS.md`](docs/KERNELS.md) — what's in the kernel corpus and how to read it.
- [`docs/BACKEND_COVERAGE.md`](docs/BACKEND_COVERAGE.md) — generated backend coverage matrix.
- [`docs/KERNEL_INDEX.md`](docs/KERNEL_INDEX.md) — generated domain-grouped kernel index.
- [`docs/PUBLIC_PRIVATE_BOUNDARY.md`](docs/PUBLIC_PRIVATE_BOUNDARY.md) — what is public, private, and generated.
- [`docs/SBOM.md`](docs/SBOM.md) — dependency and license summary for the release tree.
- [`docs/PYPI_RELEASE.md`](docs/PYPI_RELEASE.md) — TestPyPI/PyPI release procedure.
- [`docs/ALPHA_RELEASE_CHECKLIST.md`](docs/ALPHA_RELEASE_CHECKLIST.md) — release-readiness checklist.
- [`SECURITY.md`](SECURITY.md) — vulnerability reporting policy.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). External contributions to the
hand-authored layers — runtime, examples, docs, tests, and tooling — are
welcome through the normal pull-request flow. The kernel sources are
regenerated as a unit; the contribution model for those paths is documented
in CONTRIBUTING.md.

## Provenance

The kernel sources under `kernels/{cuda,hip,opencl,vulkan,metal}/` and the
generated Python facade modules under `python/navatala_gpu/` are produced
from an upstream specification and regenerated together per release. The
`kernels/manifest.json` file is the machine-readable provenance record;
[docs/KERNEL_INDEX.md](docs/KERNEL_INDEX.md) and
[docs/BACKEND_COVERAGE.md](docs/BACKEND_COVERAGE.md) are rendered from it.
See [CONTRIBUTING.md](CONTRIBUTING.md) for how patches against these paths
are routed.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

```
Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
```
