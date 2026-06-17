# Backend Coverage

Source-file availability across the five backends in this package.
Backend presence is a packaging fact — it does not imply numerical
validation, which is tracked separately per backend.

| Backend | Kernels | Source files | Binary artifacts | Coverage vs CUDA | Missing vs CUDA |
|---------|--------:|-------------:|-----------------:|-----------------:|----------------:|
| CUDA | 2,080 | 2,080 | 0 | 100.0% | 0 |
| HIP | 2,080 | 2,080 | 0 | 100.0% | 0 |
| OPENCL | 1,942 | 1,942 | 0 | 93.4% | 138 |
| VULKAN | 1,928 | 1,928 | 1,928 | 92.7% | 152 |
| METAL | 1,420 | 1,420 | 0 | 68.3% | 660 |

Notes:

- Vulkan has both GLSL compute sources (`.comp`) and generated SPIR-V
  binaries (`.spv`); the table counts those separately.
- Metal coverage is lower primarily because Apple GPUs generally do not
  support double-precision compute.
- OpenCL and Vulkan miss a small number of kernels whose atomics, BF16/F16,
  random-state, or F64 requirements do not currently map cleanly.
- Backend presence is not the same as benchmarked performance or full
  application-level validation.
