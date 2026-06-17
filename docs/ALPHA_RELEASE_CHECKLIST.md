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
- [ ] Numerical conformance status is documented separately from source coverage.
- [x] Known limitations are listed before tag.

## Release Hygiene

- [x] `LICENSE`, `NOTICE`, and root `SECURITY.md` are present.
- [x] SBOM/license summary is present under `docs/SBOM.md`.
- [x] No private specification sources or generator implementation are copied.
- [x] No absolute local paths, credentials, or internal hostnames are present.
- [x] Public/private boundary is reviewed before publication.

