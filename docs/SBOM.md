# Dependency And License Summary

This document is the release-time dependency and license summary for the
`navatala_gpu` developer-preview source tree. It is intentionally conservative:
optional GPU SDKs are build-time/runtime dependencies selected by the user and
are not redistributed by this source package.

## Project License

- Package: `navatala_gpu`
- License: Apache License 2.0
- Copyright: `Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd`

## Redistributed Source Components

| Component | Location | License | Notes |
| --- | --- | --- | --- |
| Navatala GPU runtime | `runtime/` | Apache-2.0 | Public C++20 runtime and examples. |
| Kernel corpus | `kernels/` | Apache-2.0 | Backend kernel sources projected into the release tree. |
| Python package sources | `python/` | Apache-2.0 | Python facades, type stubs, tests, and nanobind extension source. |
| DLPack header license | `runtime/include/dlpack/LICENSE` | Apache-2.0 | Included only where the DLPack header is present in the runtime tree. |

## Optional External Dependencies

These dependencies are detected or linked only when the corresponding backend or
package feature is enabled by the user. They are not vendored in this source
package.

| Dependency | Purpose | License / Terms |
| --- | --- | --- |
| NVIDIA CUDA Toolkit / Driver API | CUDA backend compilation and execution | NVIDIA Software License Agreement |
| AMD ROCm / HIP | HIP backend compilation and execution | MIT and component-specific ROCm licenses |
| Khronos OpenCL headers + ICD loader | OpenCL backend compilation and execution | Apache-2.0 / vendor ICD terms |
| Vulkan SDK, glslang, shader tools | Vulkan compute backend and SPIR-V tooling | Apache-2.0 / BSD-style component licenses |
| Apple Metal framework | Metal backend on macOS | Apple platform SDK terms |
| CMake | C++ build system | BSD-3-Clause |
| Python, scikit-build-core, nanobind | Python wheel build and bindings | Python Software Foundation License / Apache-2.0 / BSD-style licenses |
| pytest, numpy | Optional tests and examples | MIT / BSD-style licenses |

## Scan Evidence

Before tagging a public release, run the release hygiene checks from the root of
the projected tree:

```bash
tools/check_generated_content_tells.sh .
python tools/check_release_hygiene.py .
```

For distribution builds, also run package-level checks:

```bash
cmake -S . -B build-runtime \
  -DNAVATALA_GPU_USE_CUDA=OFF \
  -DNAVATALA_GPU_USE_HIP=OFF \
  -DNAVATALA_GPU_USE_VULKAN=OFF \
  -DNAVATALA_GPU_USE_OPENCL=OFF \
  -DNAVATALA_GPU_USE_METAL=OFF \
  -DNAVATALA_GPU_BUILD_TESTS=OFF
cmake --build build-runtime -j

cd python
python -m build \
  -Ccmake.define.NAVATALA_FFI_INCLUDE_DIR="$PWD/../runtime/include" \
  -Ccmake.define.NAVATALA_FFI_LIBRARY="$PWD/../build-runtime/runtime/libgpu_runtime.so"
twine check dist/*
```

The formal SBOM artifact for release tags should be generated from the final
source tree and wheel contents using a standard SPDX 2.3 or CycloneDX tool.
This source document records the intended dependency boundary and license
positioning for the developer-preview tree; it does not replace tag-specific
scanner output.
