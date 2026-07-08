# Backend Coverage

Source-file availability across the five backends in this package.
Backend presence is a packaging fact — it does not imply numerical
validation, which is tracked separately per backend.

| Backend | Kernels | Source files | Binary artifacts | Coverage vs CUDA | Missing vs CUDA |
|---------|--------:|-------------:|-----------------:|-----------------:|----------------:|
| CUDA | 2,114 | 2,114 | 0 | 100.0% | 0 |
| HIP | 2,128 | 2,128 | 0 | 100.7% | 0 |
| OPENCL | 1,976 | 1,976 | 0 | 93.5% | 138 |
| VULKAN | 1,962 | 1,962 | 1,962 | 92.8% | 152 |
| METAL | 1,444 | 1,444 | 0 | 68.3% | 670 |

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
