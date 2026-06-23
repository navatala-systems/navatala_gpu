# Navatala GPU 0.1.2 Alpha Candidate Evidence

Candidate tag: `v0.1.2-alpha.1`

This document records the locally reproducible release gates for the 0.1.2
developer-preview alpha candidate. Hardware-backed backend maturity reports
remain separate artifacts and are not implied by this local evidence.

## Candidate Scope

- Public runtime and wrapper headers.
- Generated five-backend kernel source corpus.
- Generated host registry sources.
- Python package scaffold and nanobind extension.
- Public examples and no-GPU conformance tests.

## Local Gates Run

The private release build produced a fresh public tree with CUDA, HIP, Metal,
Vulkan, and OpenCL source coverage. The following checks were then run against
the public tree:

```bash
tools/check_generated_content_tells.sh .
python3 tools/check_release_hygiene.py .
```

No-backend CMake/CTest gate:

```bash
rm -rf /tmp/navatala_gpu_conformance_build
cmake -S . -B /tmp/navatala_gpu_conformance_build \
  -DNAVATALA_GPU_USE_CUDA=OFF \
  -DNAVATALA_GPU_USE_HIP=OFF \
  -DNAVATALA_GPU_USE_VULKAN=OFF \
  -DNAVATALA_GPU_USE_OPENCL=OFF \
  -DNAVATALA_GPU_USE_METAL=OFF
cmake --build /tmp/navatala_gpu_conformance_build -j
ctest --test-dir /tmp/navatala_gpu_conformance_build --output-on-failure
```

Result: `13/13` tests passed, including:

- runtime no-backend tests;
- C ABI example;
- C++ wrapper AXPY example;
- C++ wrapper CSR-SpMV example;
- registry partition lookup example;
- `manifest_conformance` public manifest/file coverage test.

Python package gate:

```bash
cd navatala_gpu/python
rm -rf build dist
python3 -m build --sdist --wheel
python3 -m twine check dist/*
```

Result:

- `navatala_gpu-0.1.2.tar.gz` built and passed `twine check`;
- `navatala_gpu-0.1.2-cp313-cp313-linux_x86_64.whl` built locally and
  passed `twine check`;
- fresh-venv import smoke returned version `0.1.2`, ABI `6`, manifest
  `pyabi6-2675513039`, and 37 Python operation descriptors.

## Explicit Deferred Items

The following are not part of the local alpha gate:

- manylinux_2_28 x86_64/aarch64 and macOS arm64 release wheel matrix;
- ROCm/HIP CI runner and benchmark report;
- rocBLAS/rocSPARSE/hipSPARSELt vendor-dispatch paths;
- CUDA/HIP/Metal broad numerical conformance reports;
- full host-registry aggregator as a stable public CMake target;
- `openfoam_gpu` co-release readiness.

## Release Interpretation

The alpha candidate is suitable for source inspection, no-GPU packaging
validation, public runtime/wrapper integration experiments, and backend-source
coverage review. It is not a claim of complete vendor-library parity or
production numerical certification across all backend/device pairs.
