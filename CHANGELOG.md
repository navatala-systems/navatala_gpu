# Changelog

All notable user-visible changes to `navatala_gpu` are recorded here. This
file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Pre-alpha (unreleased)

The initial public developer-preview release. The package is pre-1.0; the
public surface — module names, C ABI symbols, file layout — may change in
any 0.x release until the first stable cut.

### Public surface

- Curated Python modules: `navatala_gpu.runtime`, `navatala_gpu.linalg`,
  `navatala_gpu.sparse`, `navatala_gpu.dataframe`, `navatala_gpu.graph`,
  `navatala_gpu.ml`, `navatala_gpu.cfd`. The `navatala_gpu.kernels` module
  is intentionally `internal` — see `docs/PUBLIC_PRIVATE_BOUNDARY.md`.
- C ABI: `navatala/navatala_ffi.h` is the single public C API surface.
- Five GPU backends: CUDA, HIP, OpenCL, Vulkan (GLSL compute + SPIR-V),
  Metal. Per-backend coverage is not uniform — see
  `docs/BACKEND_COVERAGE.md`.

### BREAKING — kernel symbol and filename rename

The C ABI symbol naming and on-disk kernel filenames moved to a uniform
`navatala_<module>_<op>_<dtype>` shape. There is no compatibility shim;
consumers that pinned old symbol names need to update.

The C ABI revision in `navatala/navatala_ffi.h` (`NAVATALA_GPU_FFI_ABI_VERSION`)
is bumped from `5` to `6` in lockstep with this rename. `pip`-installed
wheels will report `navatala_gpu.__abi_version__ == 6`.

Representative before/after pairs:

| Old C symbol | New C symbol |
| --- | --- |
| legacy dense GEMM symbol | `navatala_linalg_gemm_f32` |
| legacy Lanczos symbol | `navatala_sparse_lanczos_step_f32` |
| legacy KNN prediction symbol | `navatala_ml_knn_predict_f32` |
| legacy histogram symbol | `navatala_dataframe_histogram_i32` |
| legacy graph embedding symbol | `navatala_graph_spectral_embed_f32` |
| legacy CFD predictor symbol | `navatala_cfd_predictor_hby_a_f32` |
| legacy pressure-solver symbol | `navatala_cfd_poisson_pcg_f64` |
| legacy radius-search symbol | `navatala_cfd_radius_search_f32` |
| legacy top-k symbol | `navatala_dataframe_top_k_select_i32` |
| legacy eigen solver symbol | `navatala_linalg_jacobi_eigen_f64` |
| legacy AMG strength symbol | `navatala_sparse_strength_affinity_f32` |
| legacy clustering symbol | `navatala_ml_dbscan_fit_f32` |
| legacy k-means symbol | `navatala_ml_kmeans_plus_plus_init_f32` |
| legacy random generator symbol | `navatala_dataframe_uniform_lcg_f32` |
| legacy scan symbol | `navatala_dataframe_inclusive_scan_f64` |
| legacy norm symbol | `navatala_linalg_norm_f32` |
| legacy attention-softmax symbol | `navatala_ml_attention_softmax_f16` |
| legacy iterative-solver symbol | `navatala_sparse_cg_iterate_f32` |
| legacy PageRank symbol | `navatala_graph_pagerank_iter_f32` |
| legacy tree-prediction symbol | `navatala_ml_tree_predict_f32` |

Migration paths:

- **Python consumers** call the curated facades (`navatala_gpu.linalg.gemm(a, b)`);
  no symbol changes are visible at the Python surface.
- **C/C++ consumers** linking the FFI directly need to update symbol names
  in their wrapper layers. The mapping table above covers ~20 of the most
  reachable names; full coverage is in `kernels/manifest.json` (post-T2).

### Kernel file layout

- Kernel files now live under `kernels/<backend>/<module>/<op>_<dtype>.<ext>`
  (e.g., `kernels/cuda/linalg/gemm_f32.cu`). Previously they were a flat list
  under `kernels/<backend>/`.
- The host-side registry source is now `runtime/include/_registry.hpp` +
  `runtime/src/_registry.cpp` (previously used an internal registry filename).
  The safety-contracts source is
  `runtime/include/_safety_contracts.hpp` + `runtime/src/_safety_contracts.cpp`
  (previously used an internal safety-contract filename).

### Python package layout

- New private module `python/navatala_gpu/_meta.py` carries the manifest id
  and the public-symbol → public contract-id mapping for support diagnostics.
  Not part of the public API.
- `__version__ = "0.1.0"` and `__abi_version__ = 6` are exposed on the
  package surface. `__navatala_ffi_abi_version__` retained as a back-compat
  alias of `__abi_version__`.
- Smoke tests moved from the older manifest-output test folder to
  `python/tests/_manifest_driven/`.

### Documentation

- `docs/KERNELS.md`, `docs/PUBLIC_PRIVATE_BOUNDARY.md`, `docs/BACKEND_COVERAGE.md`,
  `docs/KERNEL_INDEX.md`, `docs/doxygen/kernel_catalog.dox`, and the
  top-level `README.md` reworked to lead with what the library does rather
  than where it came from. Provenance disclosure moved to a single
  `## Provenance` section in the README and a footnote in
  `PUBLIC_PRIVATE_BOUNDARY.md`.

### Contribution policy

- New `Regen-Manifest-Trailer:` git trailer required for commits touching
  `kernels/`, `python/navatala_gpu/`, or `bindings/`. Local hooks under
  `tools/install_hooks.sh`; authoritative enforcement is the
  `.github/workflows/regen-trailer.yml` workflow on every PR.
