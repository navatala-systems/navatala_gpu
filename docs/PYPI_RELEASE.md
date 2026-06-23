# PyPI Release Procedure

This document tracks the Python package release path for `navatala-gpu`.
The release tree is generated from the private source workspace; do not edit
the projected `navatala_gpu/` copy of this file directly.

## Package

- Distribution name: `navatala-gpu`
- Import package: `navatala_gpu`
- Current packaging version: `0.1.2`
- Initial PyPI version: `0.1.0` (published 2026-06-17)
- License: Apache-2.0
- Python: `>=3.9`

## Trusted Publishing

Use PyPI Trusted Publishing / OpenID Connect rather than long-lived API
tokens.

TestPyPI pending publisher:

- Project: `navatala-gpu`
- Owner: `navatala-systems`
- Repository: `navatala_gpu`
- Workflow: `publish-pypi.yml`
- Environment: `testpypi`

PyPI pending publisher:

- Project: `navatala-gpu`
- Owner: `navatala-systems`
- Repository: `navatala_gpu`
- Workflow: `publish-pypi.yml`
- Environment: `pypi`

Recommended GitHub environments:

- `testpypi`: no approval requirement; used for rehearsal publishes.
- `pypi`: require manual approval before deployment.

## Workflow

The generated workflow lives at `.github/workflows/publish-pypi.yml`.
It is manual-only:

1. Open GitHub Actions.
2. Select `Publish Python Package`.
3. Run the workflow with `target=testpypi`.
4. Install from TestPyPI and run a smoke test.
5. If the result is acceptable, run again with `target=pypi`.

The publish workflow builds and uploads `manylinux_2_28_x86_64` binary wheels
for CPython 3.9 through 3.13, plus exactly one source distribution. Wheel jobs
use `cibuildwheel` and the PyPA `manylinux_2_28` image. The source
distribution is intentionally built without `NAVATALA_FFI_*` CMake defines so
it exercises the bundled-runtime fallback path.

`auditwheel` may add an older compatible platform tag when the built extension
does not use newer glibc symbols. For example, a wheel built in the
`manylinux_2_28` container may be named with both `manylinux_2_27_x86_64` and
`manylinux_2_28_x86_64`; this is acceptable as long as the build itself ran
inside the configured `manylinux_2_28` container.

## Local Validation

First verify the standalone source distribution. This should work without any
external runtime path; the source archive carries `python/runtime/` and builds
the public runtime in stub mode:

```bash
cd python
rm -rf dist build *.egg-info
python -m build --sdist
python -m twine check dist/*.tar.gz

python -m venv /tmp/navatala_gpu_sdist_test
. /tmp/navatala_gpu_sdist_test/bin/activate
pip install --upgrade pip
pip install dist/navatala_gpu-*.tar.gz
python - <<'PY'
import navatala_gpu
print(navatala_gpu.__version__)
print(navatala_gpu.__abi_version__)
print(navatala_gpu.get_capabilities())
PY
deactivate
```

Then build local `manylinux_2_28_x86_64` wheels. This requires Docker because
`cibuildwheel` runs the build inside the PyPA manylinux container:

```bash
cd ..
rm -rf wheelhouse
python -m pip install --upgrade cibuildwheel twine
CIBW_BUILD="cp39-* cp310-* cp311-* cp312-* cp313-*" \
CIBW_ARCHS_LINUX="x86_64" \
CIBW_MANYLINUX_X86_64_IMAGE="manylinux_2_28" \
CIBW_SKIP="*musllinux*" \
CIBW_TEST_COMMAND='python -c "import navatala_gpu; assert navatala_gpu.get_capabilities().get(\"extension_loaded\")"' \
python -m cibuildwheel python --output-dir wheelhouse
python -m twine check wheelhouse/*
```

Install the built wheel into a clean environment:

```bash
python -m venv /tmp/navatala_gpu_whltest
. /tmp/navatala_gpu_whltest/bin/activate
pip install --upgrade pip
pip install wheelhouse/navatala_gpu-*-cp311-*.whl
python - <<'PY'
import navatala_gpu
print(navatala_gpu.__version__)
print(navatala_gpu.__abi_version__)
print(navatala_gpu.get_capabilities())
PY
```

## Wheel Policy

PyPI/TestPyPI reject raw `linux_x86_64` wheels. The GitHub workflow therefore
uses `cibuildwheel` with the PyPA `manylinux_2_28` image for Linux wheels
instead of building directly on the GitHub Ubuntu host. The active alpha
release target is:

- `manylinux_2_28_x86_64`

Future platform targets:

- `manylinux_2_28_aarch64`
- `macosx_11_0_arm64`

For `manylinux_2_28_aarch64`, use either native GitHub ARM runners or QEMU
via `docker/setup-qemu-action`; QEMU is slower and should be tested separately
before adding it to the publish workflow. For macOS ARM, add a separate
`macos-14` cibuildwheel job once the Metal/runtime packaging path is ready for
public wheels.

## TestPyPI Install

After the TestPyPI publish completes:

```bash
python -m venv /tmp/navatala_gpu_testpypi
. /tmp/navatala_gpu_testpypi/bin/activate
pip install --upgrade pip
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  navatala-gpu
python - <<'PY'
import navatala_gpu
print(navatala_gpu.__version__)
print(navatala_gpu.__abi_version__)
print(navatala_gpu.get_capabilities())
PY
```

Use `--extra-index-url` because TestPyPI may not carry transitive
dependencies such as NumPy.
