<!-- SPDX-License-Identifier: Apache-2.0 -->

# navatala_gpu

Generated Python bindings and runtime facade for Navatala GPU operations.

## Scope

- This package contains Python modules, a nanobind extension scaffold, machine-readable API metadata, and smoke tests for the curated public surface.
- The package links against the public Navatala GPU C ABI exposed by `navatala/navatala_ffi.h`.
- Raw kernel wrappers under `navatala_gpu.kernels` are an internal/advanced surface; most users should prefer the curated modules such as `navatala_gpu.runtime` and `navatala_gpu.linalg`.

## Build

The default package build uses the bundled public runtime in stub mode, so
`pip install .` works on hosts without GPU SDKs. Advanced users may link
against an already-built runtime by passing the public runtime include
directory and shared library:

```bash
cmake -S . -B build \
  -DNAVATALA_FFI_INCLUDE_DIR=/path/to/gpu_runtime/include \
  -DNAVATALA_FFI_LIBRARY=/path/to/libgpu_runtime.so
cmake --build build -j
```

For Python packaging, use a build frontend compatible with `scikit-build-core`.

## Python Usage

```python
import navatala_gpu

caps = navatala_gpu.get_capabilities()
assert "extension_loaded" in caps
assert isinstance(navatala_gpu.supports("axpy", backend="opencl", dtype="float32"), bool)
```

## Runtime Notes

- `import navatala_gpu` is designed to be safe on hosts without GPU drivers.
- Backend availability is reported through `navatala_gpu.get_capabilities()` and `navatala_gpu.supports(...)`.
- NumPy interop uses explicit host/device copies through `Tensor.from_numpy(...)` and `Tensor.to_numpy(...)`.
- DLPack interop is zero-copy only where the runtime can prove backend compatibility.

Manifest id: `pyabi6-3906333046`
