# Backend Coverage

Source-file availability across the five backends in this package.
Backend presence is a packaging fact — it does not imply numerical
validation, which is tracked separately per backend.

| Backend | Kernels | Source files | Binary artifacts | Coverage vs CUDA | Missing vs CUDA |
|---------|--------:|-------------:|-----------------:|-----------------:|----------------:|
| CUDA | 2,083 | 2,083 | 0 | 100.0% | 0 |
| HIP | 2,093 | 2,093 | 0 | 100.5% | 0 |
| OPENCL | 1,945 | 1,945 | 0 | 93.4% | 138 |
| VULKAN | 1,931 | 1,931 | 1,931 | 92.7% | 152 |
| METAL | 1,423 | 1,423 | 0 | 68.3% | 660 |

Notes:

- Vulkan has both GLSL compute sources (`.comp`) and generated SPIR-V
  binaries (`.spv`); the table counts those separately.
- Metal coverage is lower primarily because Apple GPUs generally do not
  support double-precision compute.
- OpenCL and Vulkan miss a small number of kernels whose atomics, BF16/F16,
  random-state, or F64 requirements do not currently map cleanly.
- Backend presence is not the same as benchmarked performance or full
  application-level validation.
- Current alpha entries are portable Navatala kernel paths unless
  `kernels/manifest.json` explicitly marks a backend path as
  vendor-backed.
