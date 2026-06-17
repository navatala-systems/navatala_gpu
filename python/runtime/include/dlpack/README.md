# Vendored DLPack Header

This directory vendors DLPack v1.0 for the `navatala_gpu` Python/DLPack bridge.

- Upstream: https://github.com/dmlc/dlpack
- Version: `v1.0`
- Header: `include/dlpack/dlpack.h`
- License: Apache-2.0, copied in `LICENSE`

The v1.0 release is used because it defines `DLManagedTensorVersioned` and
versioned DLPack flags required by `DLPACK_RUNTIME_INTEGRATION_R3.md`.
