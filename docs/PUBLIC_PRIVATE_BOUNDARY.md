# Public / Private Boundary

`navatala_gpu` is a public developer-preview package. This page describes
exactly what ships in the published distribution.

## Included In This Public Package

- `runtime/`: the cross-platform C++ GPU runtime (CUDA / HIP / OpenCL / Vulkan / Metal).
- `kernels/{cuda,hip,opencl,vulkan,metal}/`: backend kernel sources, organized by domain.
- `kernels/manifest.json`: machine-readable kernel index — backend availability, dtypes, dispatch metadata.
- `runtime/include/navatala/_registry.hpp` + `runtime/src/internal/`: host-registry sources for runtime lookup and advanced integration.
- `python/`: Python package facade and nanobind extension scaffold.
- `orchestrator/`: Volume-of-Fluid pressure-projection orchestration example.
- `examples/`: small standalone examples.
- `docs/`: public API reference, backend coverage tables, kernel index, and the release checklist.
- `tools/`: release-hygiene utilities and the trailer-policy hook installer.

## Not Included

- The internal specification source for the kernels and Python bindings.[^provenance]
- The toolchain that produces those artifacts.
- Internal discovery/orchestration pipeline scripts.
- Private project integrations and unpublished validation infrastructure.
- Private namespace paths and internal GPU IR type strings. Public manifests,
  metadata, and registry lookup paths use stable public contract ids or public
  kernel symbols instead.

[^provenance]: This is a deliberate IP/scope choice for the alpha release.
The artifacts ship as a developer-preview SDK; stable public contract ids keep
support traceability without exposing private source layout. The contribution model below
describes how patches against the regenerated paths are routed without
requiring access to the internal specification source.

## Contribution Model

Hand-authored layers — runtime, examples, docs, tests, tooling — accept
normal pull requests. The regenerated paths (kernel sources, generated Python
facade modules, nanobind bindings) follow a different flow: bug reports and
reproducers route through the maintainers, who apply the fix internally and
regenerate the public tree. The repository ships a `Regen-Manifest-Trailer:`
git hook that enforces this distinction on commits; see
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full policy.

## Maturity Label

This package should be described as a developer preview / alpha release until
the conformance suite, wrapper APIs, backend validation matrix, and selective
backend tuning work are complete.
