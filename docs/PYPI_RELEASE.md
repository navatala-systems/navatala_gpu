# PyPI Release Procedure

This document tracks the Python package release path for `navatala-gpu`.
The release tree is generated from the private source workspace; do not edit
the projected `navatala_gpu/` copy of this file directly.

## Package

- Distribution name: `navatala-gpu`
- Import package: `navatala_gpu`
- Initial version: `0.1.0`
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

The publish workflow currently builds and uploads repaired binary wheels for
CPython 3.9 through 3.13. It intentionally does **not** publish the sdist yet:
the source distribution requires additional work so it can build the bundled
runtime without external `NAVATALA_FFI_*` CMake defines.

## Local Validation

Build the runtime shared library with all optional GPU backends disabled, then
build the Python source distribution and wheel against that library:

```bash
cmake -S . -B /tmp/navatala_gpu_runtime_pypi \
  -DNAVATALA_GPU_USE_CUDA=OFF \
  -DNAVATALA_GPU_USE_HIP=OFF \
  -DNAVATALA_GPU_USE_VULKAN=OFF \
  -DNAVATALA_GPU_USE_OPENCL=OFF \
  -DNAVATALA_GPU_USE_METAL=OFF \
  -DNAVATALA_GPU_BUILD_TESTS=OFF
cmake --build /tmp/navatala_gpu_runtime_pypi -j

cd python
python -m build \
  -Ccmake.define.NAVATALA_FFI_INCLUDE_DIR="$PWD/../runtime/include" \
  -Ccmake.define.NAVATALA_FFI_LIBRARY="/tmp/navatala_gpu_runtime_pypi/runtime/libgpu_runtime.so"
python -m twine check dist/*
```

Install the built wheel into a clean environment:

```bash
python -m venv /tmp/navatala_gpu_whltest
. /tmp/navatala_gpu_whltest/bin/activate
pip install --upgrade pip
pip install python/dist/navatala_gpu-*.whl
python - <<'PY'
import navatala_gpu
print(navatala_gpu.__version__)
print(navatala_gpu.__abi_version__)
print(navatala_gpu.get_capabilities())
PY
```

## Wheel Policy

PyPI/TestPyPI reject raw `linux_x86_64` wheels. The current GitHub workflow
therefore repairs the Linux wheel with `auditwheel` before `twine check` and
upload. It publishes wheels only until the sdist can build the bundled runtime
standalone. Before a production PyPI release intended for general users,
replace the single-runner repair path with a real wheel matrix:

- `manylinux_2_28_x86_64`
- `manylinux_2_28_aarch64`
- `macosx_11_0_arm64`

Use `cibuildwheel` or an equivalent manylinux container flow so the wheel tags
match the platform policy. The existing `wheel-hygiene.yml` workflow records
the intended tags and validates package hygiene, but it is not a complete
manylinux wheel builder yet.

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
