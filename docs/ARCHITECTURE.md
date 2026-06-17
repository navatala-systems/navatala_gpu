# Architecture

Navatala GPU is intentionally two layers that are useful in isolation and
better together.

```
+-------------------------------------------------------------+
|                       Your application                     |
+-------------------------------------------------------------+
                  |                            |
                  v                            v
+---------------------------+  +------------------------------+
|       runtime/            |  |  runtime/src/internal/       |
|  cross-platform GPU C++   |  |  kernel registry + loader    |
|  abstraction (Device,     |<-+  (header at                  |
|  Queue, Buffer, Event,    |  |   runtime/include/navatala/) |
|  Program, Graph)          |  +------------------------------+
+---------------------------+
       |    |    |    |    |
       v    v    v    v    v
     CUDA  HIP  Vulkan OpenCL Metal
```

## Layer 1: the runtime

`runtime/` is a self-contained C++20 library. It compiles against any subset
of the supported backends and falls back gracefully when a backend's toolkit
isn't on the build host. Public API lives in `runtime/include/gpu_runtime.h`.

Core abstractions:

- **`Device`** — represents a physical device. Created via
  `Device::create(deviceId)`; auto-selects the first available backend (override
  with the `GPU_RUNTIME_BACKEND` environment variable).
- **`Queue`** — an execution stream. Multiple queues per device, with optional
  priority hints.
- **`Buffer`** — a typed range of memory in one of three kinds: `Device`,
  `HostPinned`, or `Managed`.
- **`Program`** — a compiled kernel module loaded from source or from the
  on-disk cache.
- **`Event`** — a synchronization primitive for cross-queue ordering, with
  optional timeout-bounded waits.
- **`Graph`** — a captured sequence of operations that can be replayed with
  reduced launch overhead (CUDA/HIP only).

Each backend lives in its own `runtime/src/backend_*.cpp` file and exposes
factory functions that the top-level `device_factory.cpp` dispatches to.

## Layer 2: the kernel corpus

`kernels/{cuda,hip,opencl,vulkan,metal}/` contain the compute kernel sources
organised by domain (`linalg/`, `sparse/`, `dataframe/`, `graph/`, `ml/`,
`cfd/`, …). The runtime treats them as opaque text and JIT-compiles them
via NVRTC / hipRTC / glslc / the OpenCL driver / Metal's runtime compiler.
Cross-backend behaviour is consistent by construction; the same operation
in CUDA, HIP, OpenCL, Vulkan compute and Metal implements the same
mathematical contract.

The host-side kernel registry (`runtime/include/navatala/_registry.hpp` +
sources under `runtime/src/internal/`) is the bridge: it names each kernel,
records its entry point and parameter shape, and feeds the runtime's
`Program::compile()` path.

Per-kernel cross-backend behaviour is consistent by construction — the same
operation in CUDA, HIP, OpenCL, Vulkan compute and Metal implements the same
mathematical contract. See [KERNELS.md](KERNELS.md) for the domains covered.

## Layer 3: the example orchestrator

`orchestrator/` ships a worked example: a Volume-of-Fluid pressure-projection
orchestrator (`Navatala::Cfd::VofPressureOrchestrator`) that wires CFD
primitives into a step-by-step host driver. Not a production solver;
useful as a reference for how the runtime, the kernel registry, and the
host code play together. The class accepts host-side mesh/time types by
reference, so it can be plugged into a CFD host build by the consumer.

## Program cache

`runtime/src/program_cache.cpp` provides a persistent on-disk cache for
compiled kernel binaries (PTX for CUDA, GCN for HIP, SPIR-V for Vulkan,
metallib for Metal). Cache directory is configured via `GPU_CACHE_DIR`;
disable via `GPU_CACHE_ENABLED=0`.

## Memory model

- **Device** memory is GPU-local; fastest, no host access.
- **HostPinned** memory is page-locked host memory directly accessible from
  the GPU; lower bandwidth than device memory but no explicit transfer.
- **Managed** memory is the unified address space on CUDA/HIP; migrated
  on-demand by the driver. Not available on Vulkan.

The capability matrix lives in the project README.

## Error handling

All failures surface as `std::runtime_error` with a descriptive message
that includes the backend-specific error code where applicable. Backend
drivers translate their native error codes via lookup tables (see
`runtime/src/backend_cuda.cpp::cudaRuntimeErrorString` for the pattern).

## What's intentionally out of scope

- This release does not bundle a high-level vendor math/DNN library. The
  kernel corpus covers BLAS-level building blocks; large numerical libraries
  belong in downstream consumers.
- The internal authoring pipeline that produces `kernels/` is private. The
  shipped artifacts are self-contained and usable on their own; provenance
  is summarised in the README's `## Provenance` section.
