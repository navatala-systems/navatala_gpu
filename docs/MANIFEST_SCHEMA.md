# Manifest Schema

Public reference for the JSON manifests emitted into the published
`navatala_gpu` tree. This document describes the stable public surface of
each file. Implementation-private records and fields that are not intended
for consumers are intentionally omitted.

Status: public manifest schema reference for the packaged tree. This copy is
projected into `navatala_gpu/docs/MANIFEST_SCHEMA.md` during release-tree
assembly so package consumers can inspect the manifest formats without
access to internal tooling.

---

## `python/release_metadata.json`

Top-level provenance / release record. Single object.

| Field | Type | Description |
| --- | --- | --- |
| `manifestId` | string | Stable regeneration identifier, shape `pyabi<N>-<hash>`. Bumps when ABI breaks or bindings change. |
| `abiVersion` | integer | C ABI revision number. Matches `NAVATALA_GPU_FFI_ABI_VERSION` in `navatala/navatala_ffi.h` and `navatala_gpu.__abi_version__`. |
| `version` | string | Public SemVer string. Matches `navatala_gpu.__version__` and the `version` field in `pyproject.toml`. |
| `packageName` | string | Distribution name on PyPI (`navatala-gpu`). |
| `generatedBy` | string | Toolchain identifier (`gpu_runtime`). |
| `bindingCount` | integer | Number of Python bindings exposed by this package. |
| `modules` | string[] | List of public Python modules in this package. |
| `symbolToContractId` | object | Map from public symbol (`<module>.<pythonName>`) to the public contract id (e.g., `linalg.axpy → navatala://python/linalg/axpy`). Same data as `navatala_gpu._meta._CONTRACTS`, in machine-readable form. |

### Example

```json
{
  "manifestId": "pyabi6-2849636658",
  "abiVersion": 6,
  "version": "0.1.0",
  "packageName": "navatala-gpu",
  "generatedBy": "gpu_runtime",
  "bindingCount": 37,
  "modules": ["linalg", "sparse", "dataframe", "graph", "ml", "cfd", "runtime", "kernels"],
  "symbolToContractId": {
    "linalg.axpy": "navatala://python/linalg/axpy",
    "linalg.gemm": "navatala://python/linalg/axpy"
  }
}
```

### Consumers

- `navatala_gpu._meta` reads `manifestId` for support diagnostics.
- CI pipelines read `version` to tag wheel artifacts.
- The `tools/check_regen_trailer.sh` hook reads `manifestId` to validate
  `Regen-Manifest-Trailer:` git-trailer values.

---

## `python/api_manifest.json`

Per-binding metadata. Single object with a `bindings` array.

| Top-level field | Type | Description |
| --- | --- | --- |
| `abiVersion` | integer | Same as `release_metadata.json`. |
| `packageName` | string | Same as `release_metadata.json`. |
| `generatedBy` | string | Same as `release_metadata.json`. |
| `bindings` | array | One entry per Python binding. |

### Binding entry shape

| Field | Type | Description |
| --- | --- | --- |
| `module` | string | Public Python module (`linalg`, `sparse`, ...). |
| `pythonName` | string | Function name as exposed (`axpy`, `gemm`, ...). |
| `name` | string | Upstream contract Name. Internal use; equals the value in `symbolToContractId` for the same key. |
| `publicName` | string | Fully qualified public Python symbol, `<module>.<pythonName>`. |
| `publicModule` | string | Public module bucket. Mirrors `module` for consumers that prefer explicit traceability fields. |
| `contractId` | string | Stable traceability name for support diagnostics. Mirrors `name`. |
| `maturity` | string | `"stable"` \| `"experimental"` \| `"internal"`. |
| `doc.summary` | string | One-line summary surfaced in `help(...)`. |
| `doc.details` | string | Extended docstring body. |
| `tensors` | array | Tensor parameters and their roles. |
| `scalars` | array | Scalar parameters. |
| `backendSupport` | array | Per-backend dtype availability. |
| `dispatchVariants` | array | For dispatch-group bindings; the concrete instantiations. |
| `syncPolicy` | object | Synchronization defaults. |
| `releasesGIL` | bool | Whether the binding releases the Python GIL on call. |
| `returnsDlpackTensor` | bool | Whether the return value is DLPack-exchangeable. |

### Traceability Fields

The T2 traceability fields are now present for Python bindings:

- `publicName` — same as `<module>.<pythonName>` joined; convenience for
  search tooling.
- `publicModule` — duplicates `module`; redundant with current shape.
- `contractId` — duplicates `name`; redundant with current shape.

Per-backend source-file paths are exposed by `kernels/manifest.json` rather
than repeated in every Python binding, because a binding may dispatch to a
library operation, dispatch group, or multiple backend-specific kernels.

---

## `kernels/manifest.json`

Kernel corpus index, emitted by the release tree builder.

The `summary` block carries:

- `kernelCount` — total kernels across all backends.
- `libraryOperationCount` — total public library-operation dispatch records.
- `domains` — kernel count per public module (`linalg`, `sparse`, ...).
- `backendCoverage` — per-backend availability metrics, drives `docs/BACKEND_COVERAGE.md`.

Each kernel entry carries:

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Public kernel identifier / preferred lookup name. |
| `entryPoint` | string | Backend entry point where the backend has a named C-style symbol. Vulkan may use `main` in the source while the public lookup name remains `name`. |
| `domain` | string | Public domain bucket such as `BLAS / dense linear algebra`, `Sparse linear algebra`, or `Volume-of-Fluid CFD`. |
| `inputs` / `outputs` | array | Sanitized argument records with `name`, public `type`, access mode, and memory space. |
| `workgroupSize` | object | Intended workgroup size `{x, y, z}`. |
| `defaultDispatch` | object | Default dispatch dimensions `{x, y, z}` when the caller uses the standard launch policy. |
| `determinism` | object | Determinism defaults and capability flags. |
| `implementation` | object | Public implementation provenance for the kernel. Current alpha values are `kind: "portable_kernel"` and `vendorBacked: false`; future vendor dispatch paths must mark this explicitly. |
| `requirements` | object | Public capability requirements such as atomics, float16/float64, subgroup operations, and shared-memory estimate. |
| `backends` | object | Per-backend availability and source/binary records. |

Each `backends.<backend>` entry carries:

| Field | Type | Description |
| --- | --- | --- |
| `available` | bool | Whether a source exists for this backend. |
| `source` | string \| null | Backend-relative source path when available. |
| `binary` | string \| null | Backend-relative binary artifact path when available, currently used for Vulkan SPIR-V. |
| `entryPoint` | string \| null | Backend entry point when available. |
| `implementation` | string | `portable_kernel`, `vendor_dispatch`, `fallback_reference`, or `unsupported`. |
| `vendorBacked` | bool | `true` only when the runtime path is explicitly backed by a vendor library. Source coverage alone must not set this flag. |

The manifest also carries a `libraryOperations` array. These entries describe
wrapper/orchestrator library dispatch, not standalone kernel files. This keeps
portable fallback kernels honest while still exposing public API paths that may
call vendor libraries such as rocBLAS/cuBLAS.

Each `libraryOperations[]` entry carries:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Opaque public traceability id for the library-operation contract. |
| `operation` | string | Stable operation class such as `blas_gemm` or `sparse_spmv`. |
| `domain` | string | Public domain bucket. |
| `inputs` / `outputs` | array | Sanitized argument records with public types/access modes. |
| `implementation` | object | Runtime dispatch metadata. |

The `implementation` object carries:

| Field | Type | Description |
| --- | --- | --- |
| `kind` | string | `library_contract`, `vendor_dispatch`, `fallback_reference`, or `unsupported`. |
| `vendorBacked` | bool | `true` only for public runtime paths that are explicitly wired to a vendor library. |
| `vendorBackends` | array | Backend names with implemented vendor dispatch. Empty when `vendorBacked` is `false`. |
| `fallback` | string \| null | Public fallback kernel id/name when a fallback is declared. |
| `publicWrappers` | array | Stable public C ABI wrapper symbols that intentionally exercise this operation. |
| `notes` | string | Human-readable scope note. |

Current alpha rule: `BLAS_GEMM` may be marked `vendor_dispatch` for CUDA/HIP
wrapper paths such as `navatala_gpu_gemm_f32`; standalone tiled GEMM kernel
files remain `portable_kernel` unless their own manifest entry says otherwise.

---

## Stability

These manifest schemas are stable within a major release. The `manifestId`
identifies the exact emitter version; consumers that bind to undocumented
fields should pin against a specific `manifestId`.

Breaking schema changes bump `abiVersion` (the cross-package ABI revision)
in lockstep with the package version. The `version` field follows SemVer
relative to the published `0.1.0` alpha baseline.

---

## Validation

the emitter is expected to validate that:

- Every entry in `symbolToContractId` has a matching binding in the `bindings`
  array of `api_manifest.json`.
- No public name (`<module>.<pythonName>`) contains private namespace,
  private IR-type, or generator-content tells.
- `abiVersion` in both manifests agrees with `NAVATALA_GPU_FFI_ABI_VERSION`
  in `navatala/navatala_ffi.h`.

The binding-manifest validator checks these rules before staging the public
tree. The `navatala_gpu/tools/check_generated_content_tells.sh` script also
enforces the public-name rule against the published tree.
