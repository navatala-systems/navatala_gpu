# SPDX-License-Identifier: Apache-2.0
import importlib

DISALLOWED_PUBLIC_NAME_FRAGMENTS = (
    "cudf", "cuml", "cuvs", "cugraph", "cudnn", "nccl",
    "raft", "rmm", "rapids", "nvidia", "amd",
)
PUBLIC_MODULES = ["navatala_gpu", "navatala_gpu.linalg", "navatala_gpu.sparse", "navatala_gpu.dataframe", "navatala_gpu.graph", "navatala_gpu.ml", "navatala_gpu.cfd", "navatala_gpu.kernels"]
PUBLIC_PATHS_AND_BINDINGS = ["README.md", "api_manifest.json", "release_metadata.json", "python/navatala_gpu/linalg.py", "python/navatala_gpu/sparse.py", "python/navatala_gpu/dataframe.py", "python/navatala_gpu/graph.py", "python/navatala_gpu/ml.py", "python/navatala_gpu/cfd.py", "python/navatala_gpu/kernels.py", "docs/api/linalg.rst", "docs/api/sparse.rst", "docs/api/dataframe.rst", "docs/api/graph.rst", "docs/api/ml.rst", "docs/api/cfd.rst", "docs/api/kernels.rst", "linalg.axpy", "linalg.axpy_dispatch", "linalg.gemm", "linalg.nrm2", "dataframe.reduce_sum", "dataframe.reduce_sum_i32_to_i64", "dataframe.reduce_min_i32", "dataframe.reduce_max_i32", "dataframe.fill_null", "dataframe.fill_null_i32", "dataframe.gather", "dataframe.gather_i32", "dataframe.scatter", "dataframe.scatter_i32", "dataframe.compact_by_mask", "dataframe.compact_by_mask_i32", "dataframe.coalesce", "dataframe.copy_if_valid", "dataframe.inclusive_scan_sum_valid_prefix", "dataframe.inclusive_scan_sum_i32_to_i64_skip_nulls", "dataframe.count_valid", "dataframe.is_valid", "dataframe.is_null", "dataframe.fill_constant", "dataframe.fill_constant_i32", "dataframe.fill_range", "dataframe.fill_validity_all_valid", "dataframe.fill_validity_all_null", "ml.mean_squared_error", "ml.root_mean_squared_error", "ml.mean_absolute_error", "ml.mean_absolute_percentage_error", "ml.r2_score", "ml.explained_variance_score", "sparse.csr_spmv", "graph.out_degree", "graph.in_degree"]

def _assert_neutral(value: str, context: str) -> None:
    lower = value.lower()
    for fragment in DISALLOWED_PUBLIC_NAME_FRAGMENTS:
        assert fragment not in lower, f"{context} contains vendor-derived fragment {fragment!r}: {value!r}"

def test_public_python_modules_and_symbols_are_vendor_neutral():
    for module_name in PUBLIC_MODULES:
        _assert_neutral(module_name.replace("navatala_gpu", "navatala"), "public module")
        module = importlib.import_module(module_name)
        names = getattr(module, "__all__", None)
        if names is None:
            names = [name for name in dir(module) if not name.startswith("_")]
        for name in names:
            _assert_neutral(str(name), f"public symbol in {module_name}")

def test_public_paths_and_binding_names_are_vendor_neutral():
    for value in PUBLIC_PATHS_AND_BINDINGS:
        _assert_neutral(value.replace("navatala_gpu", "navatala"), "public path/binding")
