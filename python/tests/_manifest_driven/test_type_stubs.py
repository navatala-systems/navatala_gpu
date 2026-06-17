# SPDX-License-Identifier: Apache-2.0
import ast
import pathlib
import pytest

PACKAGE_NAME = "navatala_gpu"
EXTENSION_MODULE = "navatala_gpu_ext"

def _package_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]

def _stub_path() -> pathlib.Path:
    return _package_root() / "bindings" / f"{EXTENSION_MODULE}.pyi"

def test_type_stub_is_valid_python_syntax():
    stub = _stub_path()
    assert stub.exists(), f"missing type stub: {stub}"
    ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub), type_comments=True)

def test_type_stub_runtime_references_are_importable():
    runtime = pytest.importorskip(f"{PACKAGE_NAME}.runtime")
    assert hasattr(runtime, "Device")
    assert hasattr(runtime, "Queue")
    assert hasattr(runtime, "Event")
    assert hasattr(runtime, "Tensor")

def test_type_stub_mypy_if_available():
    mypy_api = pytest.importorskip("mypy.api")
    stdout, stderr, status = mypy_api.run(["--ignore-missing-imports", str(_stub_path())])
    assert status == 0, stdout + stderr
