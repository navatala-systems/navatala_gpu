# SPDX-License-Identifier: Apache-2.0
import pathlib

PACKAGE_NAME = "navatala_gpu"

def _package_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]

def test_readme_exists_and_documents_public_boundary():
    readme = (_package_root() / "README.md").read_text(encoding="utf-8")
    assert "import navatala_gpu" in readme
    assert "get_capabilities" in readme

def test_readme_import_and_capability_example_executes():
    pkg = __import__(PACKAGE_NAME)
    caps = pkg.get_capabilities()
    assert isinstance(caps, dict)
    assert "extension_loaded" in caps
    assert "live_dlpack_exports" in caps
    assert isinstance(pkg.supports("axpy", backend="opencl", dtype="float32"), bool)

def test_security_and_notice_files_exist():
    root = _package_root()
    for name in ["LICENSE", "NOTICE", "SECURITY.md", "api_manifest.json", "release_metadata.json"]:
        path = root / name
        assert path.exists(), f"missing public SDK file: {name}"

def test_sphinx_docs_scaffold_exists():
    root = _package_root()
    expected = ["docs/conf.py", "docs/index.rst", "docs/api/index.rst"]
    for name in expected:
        path = root / name
        assert path.exists(), f"missing docs scaffold file: {name}"
    assert "sphinx.ext.autodoc" in (root / "docs/conf.py").read_text(encoding="utf-8")
    assert "api/index" in (root / "docs/index.rst").read_text(encoding="utf-8")
    api_index = (root / "docs/api/index.rst").read_text(encoding="utf-8")
    for module in ["linalg", "sparse", "dataframe", "graph", "ml", "cfd", "kernels"]:
        assert module in api_index
