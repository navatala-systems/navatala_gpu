# Contributing to Navatala GPU

Thank you for your interest in contributing! Please take a moment to read
this guide before opening an issue or pull request.

## What can be contributed where

The repository has hand-authored layers and regenerated layers. The two flows
are described below; the regenerated layers are covered in more detail under
[Regeneration trailer policy for generated paths](#regeneration-trailer-policy-for-generated-paths).

- **`runtime/`** is the cross-platform C++ runtime library. **Contributions
  to this layer are welcome and accepted directly via pull requests.**
- **`kernels/{cuda,hip,opencl,vulkan,metal}/`** ship as regenerated source.
  PRs that directly edit these files are not merged in place. **Bug reports
  and reproducers against them are extremely valuable** — please file them
  with the kernel name, backend, and a minimal reproducer; the fix is then
  applied upstream and regenerated into the public tree.
- **`orchestrator/`** carries example host orchestrator code. Open an issue
  first to discuss scope before submitting a PR.

## Reporting bugs

When filing a bug, please include:

1. The backend(s) and device(s) you reproduced on (`device->getName()` /
   `device->getTotalMemory()` output is ideal).
2. CMake configuration: which `GPU_RUNTIME_USE_*` flags, which toolkit
   versions.
3. A minimal failing program, or, if the bug is in a generated kernel,
   the kernel file name and the input that triggers the failure.
4. Whether the failure reproduces in the runtime tests under
   `runtime/tests/`.

## Regeneration trailer policy for generated paths

**Files under `kernels/`, `python/navatala_gpu/`, and `bindings/` are regenerated
from upstream specifications.** Patches against these paths are routed through
the spec layer rather than merged in place. CI and the local git hooks reject
commits that touch these paths unless the commit message includes a
`Regen-Manifest-Trailer:` git trailer naming the manifest id the patch was
produced from (the same id present in `python/release_metadata.json` and in
`python/navatala_gpu/_meta.py`).

Install the local hooks once after cloning:

```
./tools/install_hooks.sh
```

The hooks live in `tools/check_regen_trailer.sh` (path-and-message check) and are
enforced server-side by `.github/workflows/regen-trailer.yml`. Local hooks are
best-effort because `git commit --no-verify` can bypass them; CI is the
authoritative gate.

## Pull requests

- Target the `main` branch.
- One logical change per PR — small focused PRs review fast.
- Include or update unit tests under `runtime/tests/` for runtime changes.
- Match the surrounding code style; this is C++20 with `clang-format`-style
  formatting.
- All source files must carry the SPDX Apache 2.0 header. If you add a new
  file, copy the header from any existing file in `runtime/src/`.

## License

By submitting a contribution you agree it will be released under the Apache
License 2.0 (see `LICENSE`). Please do not include code under incompatible
licenses without prior discussion.

## Code of conduct

Be kind, be specific, be patient. This is a small project; we will respond
as soon as we can.
