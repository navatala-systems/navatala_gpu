# Alpha Release Checklist

This checklist defines the minimum bar for a `navatala_gpu` developer-preview
release. It is intentionally conservative: source coverage, numerical
validation, and performance claims are tracked separately.

## Build And Packaging

- [x] Clean configure/build with all optional GPU backends disabled.
- [x] `NAVATALA_GPU_BUILD_TESTS=OFF` suppresses runtime test targets.
- [x] Optional targets are either buildable or not advertised.
- [x] Examples build with default CMake options.

## Kernel Corpus

- [x] `kernels/manifest.json` is projected into the release tree.
- [x] `docs/BACKEND_COVERAGE.md` is rendered from the manifest.
- [x] `docs/KERNEL_INDEX.md` is rendered from the manifest.
- [x] Missing backend coverage is explicit and explained.

## Documentation

- [x] README states developer-preview maturity.
- [x] Python SDK docs/API stubs are present when the Python bindings ship.
- [x] Doxygen config builds runtime/wrapper API docs.
- [x] Public docs avoid claims of full backend parity unless manifest-backed.

## Validation

- [x] No-GPU test subset passes.
- [x] Python package builds with nanobind against the public C ABI in local
      no-backend validation.
- [x] Python package tests pass against the built runtime in local no-backend
      validation.
- [ ] Release-tag manylinux/macOS wheel matrix has run and archived artifacts.
- [x] At least one GPU backend smoke test is recorded.
- [x] Numerical conformance status is documented separately from source coverage.
- [x] Known limitations are listed before tag.

## Release Hygiene

- [x] `LICENSE`, `NOTICE`, and root `SECURITY.md` are present.
- [x] SBOM/license summary is present under `docs/SBOM.md`.
- [x] No private specification sources or generator implementation are copied.
- [x] No absolute local paths, credentials, or internal hostnames are present.
- [x] Public/private boundary is reviewed before publication.

## Tracker Boundary

- [x] #1362 remains the parent developer-preview release-readiness tracker.
- [x] #1368 stable C++ wrapper API baseline is resolved for the initial
      AXPY/GEMM/NRM2/CSR-SpMV facade scope; broader AMG/CFD/ML facades are
      future API expansion.
- [x] #1373 examples/quickstart scope is closed: C ABI AXPY, C++ AXPY,
      C++ CSR SpMV, and host-registry partition lookup examples build from
      the public tree.
- [x] #1506 NumPy/DLPack Python binding baseline is resolved for the local
      no-backend build/package/import path.
- [x] #1515 subset DNN graph executor baseline is resolved, but cuDNN parity
      claims remain intentionally constrained by `docs/KERNELS.md` and
      `docs/NUMERICAL_CONFORMANCE.md`.
- [ ] #1516-#1521 ROCm/HIP CI, vendor-dispatch integration, and published
      HIP benchmark reports require AMD-capable runners or rental hardware and
      are post-alpha blockers unless a ROCm-backed release is claimed.
- [ ] #1363/#1379 `openfoam_gpu` co-release is explicitly outside this
      `navatala_gpu` alpha unless those separate CFD release-readiness tasks
      are completed. The public `orchestrator/` code is a non-turnkey example
      boundary, not an OpenFOAM solver maturity claim.

## Candidate Marker

- [ ] Record the final public repository commit SHA and tag here before
      cutting a release candidate.
- [ ] Archive the exact `tools/check_generated_content_tells.sh` and
      `tools/check_release_hygiene.py` outputs for the candidate commit.
- [ ] Archive the wheel/sdist artifact list and platform tags for the candidate
      commit.

## Deferred Post-Alpha Items

- [ ] manylinux_2_28 x86_64/aarch64 and macOS arm64 release wheels must be
      built in the release workflow and archived before a production release.
- [ ] CUDA/HIP/Metal hardware correctness and performance reports must be
      attached before claiming backend maturity beyond developer preview.
- [ ] Full host-registry aggregator should get a dedicated public CMake target
      before advertising it as a stable linkable runtime component. The current
      example intentionally verifies a generated registry partition only.
- [ ] `openfoam_gpu` release readiness remains tracked separately under
      #1363/#1379 and related CFD tasks.

